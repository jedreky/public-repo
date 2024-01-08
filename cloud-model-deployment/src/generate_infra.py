from enum import Enum, auto
import json
import logging
from pathlib import Path
import yaml

from aws_cdk import App, Stack, Fn
import aws_cdk as cdk
import aws_cdk.aws_autoscaling as autoscaling
import aws_cdk.aws_ec2 as ec2
import aws_cdk.aws_elasticloadbalancingv2 as elbv2
import aws_cdk.aws_elasticloadbalancingv2_targets as targets
import aws_cdk.aws_iam as iam
import aws_cdk.aws_imagebuilder as imagebuilder

from src.config import EXTERNAL_SSH_KEY, INTERNAL_SSH_KEY, WHISPER_AMIS

from src.constants import (
    HTTP_PORT,
    SSH_PORT,
    DEFAULT_REGION,
    EC2_USER,
    EC2_HOME_FOLDER,
    BUILD_INSTANCE_TYPES,
    CLIENT_INSTANCE_TYPE,
    REPO_NAME,
    FOLDER_NAME,
)

MAIN_FOLDER = Path(__file__).parent.parent
CFN_TEMPLATES_FOLDER = MAIN_FOLDER / "cfn_templates"
USER_DATA_FOLDER = MAIN_FOLDER / "user_data"

for folder in [CFN_TEMPLATES_FOLDER, USER_DATA_FOLDER]:
    if not folder.exists():
        folder.mkdir()


class InstanceRole(Enum):
    WORKER = auto()
    CLIENT = auto()


def clean_template(template):
    bits_to_remove = {
        "Parameters": ["BootstrapVersion"],
        "Rules": ["CheckBootstrapVersion"],
    }
    for key, val in bits_to_remove.items():
        for x in val:
            if x in template[key]:
                template[key].pop(x)
                logging.debug(f"Removing {x} from {key}")

    for key, val in template["Resources"].items():
        if "Metadata" in val:
            val.pop("Metadata")

        if "Tags" in val["Properties"]:
            val["Properties"].pop("Tags")


def save_templates(cloud_assembly):
    for stack in cloud_assembly.stacks:
        template = stack.template
        clean_template(template)

        with open(CFN_TEMPLATES_FOLDER / f"{stack.display_name}.json", "w") as f:
            json.dump(template, f, indent=2)


def render_code(code):
    code_str = ""

    for line in code:
        if line.startswith("# "):
            code_str += f"\n{line}\n"
        else:
            code_str += f"{line}\n"

    return code_str


def get_ami_type(instance_type):
    if "g." in instance_type:
        ret = ("arm",)
    else:
        ret = ("x86",)

    if instance_type.startswith("g"):
        ret += ("gpu",)
    else:
        ret += ("cpu",)

    return ret


def get_user_data(role, *, details=None):
    user_data = [
        "#!/bin/bash -xe",
        f"cd {EC2_HOME_FOLDER}",
    ]

    if role == InstanceRole.CLIENT:
        user_data = []

        if (MAIN_FOLDER / "keys" / "internal_ssh_key.pem").exists():
            with open(MAIN_FOLDER / "keys" / "internal_ssh_key.pem") as f:
                internal_ssh_key = f.read()

            user_data += [
                f'echo "{internal_ssh_key}" > .ssh/id_ed25519',
                "chmod 400 .ssh/id_ed25519",
                f"chown {EC2_USER}:{EC2_USER} .ssh/id_ed25519",
            ]

        endpoint = details["endpoint"]
        user_data += [
            f"echo 'export ENDPOINT={endpoint}' >> .bash_profile",
            "# disable and stop whisper service just in case",
            "systemctl disable whisper",
            "systemctl stop whisper",
        ]

    if role == InstanceRole.WORKER:
        user_data += [
            "# enable and restart whisper service",
            "systemctl enable whisper",
            "systemctl start whisper",
        ]

    user_data_str = render_code(user_data)
    filename = f"{role.name.lower()}.sh"

    with open(USER_DATA_FOLDER / filename, "w") as f:
        f.write(user_data_str)

    return user_data_str


class BaseStack(Stack):
    def __init__(
        self,
        scope,
        construct_id,
        *,
        single_subnet,
        num_azs,
        region=DEFAULT_REGION,
        debug_mode=False,
    ):
        super().__init__(scope, construct_id)

        self.current_region = region
        subnet_configuration = [
            ec2.SubnetConfiguration(
                subnet_type=ec2.SubnetType.PUBLIC, name="PublicSubnet", cidr_mask=24
            )
        ]

        if not single_subnet:
            subnet_configuration.append(
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    name="PrivateSubnet",
                    cidr_mask=24,
                )
            )

        self.vpc = ec2.Vpc(
            self,
            "VPC",
            ip_addresses=ec2.IpAddresses.cidr("10.0.0.0/16"),
            restrict_default_security_group=False,
            max_azs=num_azs,
            subnet_configuration=subnet_configuration,
        )
        self.ssh_inbound_security_group = ec2.SecurityGroup(
            self,
            "SSHInboundSecurityGroup",
            vpc=self.vpc,
            description=f"Allow inbound SSH ({SSH_PORT}), allow all outbound",
            allow_all_outbound=True,
            disable_inline_rules=False,
        )
        self.ssh_inbound_security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(SSH_PORT)
        )

        self.http_inbound_security_group = ec2.SecurityGroup(
            self,
            "HTTPInboundSecurityGroup",
            vpc=self.vpc,
            description=f"Allow inbound HTTP ({HTTP_PORT}), allow all outbound",
            allow_all_outbound=True,
            disable_inline_rules=False,
        )
        self.http_inbound_security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(HTTP_PORT)
        )

        if debug_mode:
            self.http_inbound_security_group.add_ingress_rule(
                ec2.Peer.any_ipv4(), ec2.Port.tcp(SSH_PORT)
            )


def get_component_from_steps(scope, component_name, steps):
    component_data = {
        "name": component_name,
        "schemaVersion": 1.0,
        "phases": [
            {
                "name": "build",
            }
        ],
    }
    component_data["phases"][0]["steps"] = [
        {
            "name": name,
            "action": "ExecuteBash",
            "inputs": {"commands": commands},
        }
        for name, commands in steps
    ]

    component = imagebuilder.CfnComponent(
        scope,
        component_name,
        name=component_name,
        platform="Linux",
        version="1.0.0",
        data=yaml.dump(component_data),
    )
    return component


def get_init_component(scope):
    steps = [("Init", ["timedatectl set-timezone Europe/Warsaw", "dnf -y update"])]
    return get_component_from_steps(scope, "InitComponent", steps)


def get_cuda_component(scope):
    steps = [
        (
            "InstallCUDA",
            [
                "dnf -y install kernel-devel-$(uname -r) kernel-headers-$(uname -r)",
                "dnf -y install kernel-modules-extra.x86_64",
                "dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo",
                "dnf clean all",
                "dnf -y module install nvidia-driver:latest-dkms",
                "dnf -y install cuda-toolkit",
            ],
        )
    ]
    return get_component_from_steps(scope, "CUDAComponent", steps)


def get_main_component(scope, arch):
    if arch == "x86":
        ffmpeg_arch = "i686"
    elif arch == "arm":
        ffmpeg_arch = "arm64"

    steps = [
        (
            "InstallFFmpeg",
            [
                f"mkdir {EC2_HOME_FOLDER}/ffmpeg",
                f"cd {EC2_HOME_FOLDER}/ffmpeg",
                f"wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-{ffmpeg_arch}-static.tar.xz",
                f"tar -xf ffmpeg-release-{ffmpeg_arch}-static.tar.xz",
                f"ln -s {EC2_HOME_FOLDER}/ffmpeg/ffmpeg-*-{ffmpeg_arch}-static/ffmpeg /usr/bin/ffmpeg",
            ],
        ),
        (
            "SetupRepo",
            [
                "dnf -y install git python3.11",
                "ln -s /usr/bin/python3.11 /usr/bin/python",
                # f"mkdir {EC2_HOME_FOLDER}/tmp",
                # f"export TMPDIR={EC2_HOME_FOLDER}/tmp",
                f"cd {EC2_HOME_FOLDER}",
                f"git clone https://github.com/jedreky/{REPO_NAME}.git",
                f"cd {REPO_NAME}/{FOLDER_NAME}",
                "cp whisper.service /etc/systemd/system",
                "python -m venv venv",
                "source venv/bin/activate",
                "pip install --upgrade pip",
                "pip install -r requirements.txt",
                f"XDG_CACHE_HOME=/home/{EC2_USER}/.cache python -m src.initialise_model",
            ],
        ),
        (
            "FinalCleanup",
            [
                f"cd {EC2_HOME_FOLDER}",
                f'echo "if [ -z \$VIRTUAL_ENV ]; then source {EC2_HOME_FOLDER}/{REPO_NAME}/{FOLDER_NAME}/venv/bin/activate; fi" >> .bash_profile',  # noqa: W605
                f'echo "cd {EC2_HOME_FOLDER}/{REPO_NAME}/{FOLDER_NAME}" >> .bash_profile',
                f"chown -R {EC2_USER}:{EC2_USER} {EC2_HOME_FOLDER}",
            ],
        ),
    ]

    return get_component_from_steps(scope, f"MainComponent{arch.capitalize()}", steps)


class BuildStack(BaseStack):
    def __init__(
        self,
        scope,
        construct_id,
        *,
        volume_size,
        include_arm=False,
        debug_mode=False,
    ):
        super().__init__(
            scope, construct_id, single_subnet=True, num_azs=1, debug_mode=debug_mode
        )
        cpu_types = ["x86"]

        if include_arm:
            cpu_types.append("arm")

        init_component = get_init_component(self)
        cuda_component = get_cuda_component(self)
        main_component = {x: get_main_component(self, x) for x in cpu_types}

        managed_policies = [
            iam.ManagedPolicy.from_aws_managed_policy_name(x)
            for x in [
                "AmazonSSMManagedInstanceCore",
                "EC2InstanceProfileForImageBuilder",
            ]
        ]

        role = iam.Role(
            self,
            "Role",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=managed_policies,
        )
        instance_profile = iam.InstanceProfile(self, "InstanceProfile", role=role)

        base_amis = {
            x: ec2.MachineImage.latest_amazon_linux2023(
                cpu_type=getattr(ec2.AmazonLinuxCpuType, f"{x.upper()}_64")
            )
            .get_image(self)
            .image_id
            for x in cpu_types
        }

        for cpu_type in cpu_types:
            for x in ["cpu", "gpu"]:
                match x:
                    case "cpu":
                        components = [init_component, main_component[cpu_type]]
                    case "gpu":
                        components = [
                            init_component,
                            cuda_component,
                            main_component[cpu_type],
                        ]
                    case _:
                        raise RuntimeError("Unknown type!")

                suffix = x.upper() + cpu_type.capitalize()
                suffix_snake = f"{x}_{cpu_type}"
                image_name = f"Whisper{suffix}"
                image_recipe = imagebuilder.CfnImageRecipe(
                    self,
                    f"{image_name}Recipe",
                    components=[
                        imagebuilder.CfnImageRecipe.ComponentConfigurationProperty(
                            component_arn=x.attr_arn
                        )
                        for x in components
                    ],
                    name=image_name,
                    parent_image=base_amis[cpu_type],
                    version="1.0.0",
                    block_device_mappings=[
                        imagebuilder.CfnImageRecipe.InstanceBlockDeviceMappingProperty(
                            device_name="/dev/xvda",
                            ebs=imagebuilder.CfnImageRecipe.EbsInstanceBlockDeviceSpecificationProperty(
                                delete_on_termination=True, volume_size=volume_size
                            ),
                        )
                    ],
                )

                infrastructure_configuration = (
                    imagebuilder.CfnInfrastructureConfiguration(
                        self,
                        f"InfrastructureConfiguration{suffix}",
                        instance_profile_name=instance_profile.instance_profile_name,
                        name=f"InfrastructureConfiguration{suffix}",
                        instance_types=[BUILD_INSTANCE_TYPES[(cpu_type, x)]],
                        key_pair=EXTERNAL_SSH_KEY,
                        security_group_ids=[
                            self.ssh_inbound_security_group.security_group_id
                        ],
                        subnet_id=self.vpc.public_subnets[0].subnet_id,
                        terminate_instance_on_failure=False,
                    )
                )

                distribution = (
                    imagebuilder.CfnDistributionConfiguration.DistributionProperty(
                        region=self.current_region,
                        ami_distribution_configuration={
                            "Name": f"whisper_{suffix_snake}_"
                            + "{{imagebuilder:buildDate}}"
                        },
                    )
                )
                distribution_configuration = imagebuilder.CfnDistributionConfiguration(
                    self,
                    f"DistributionConfiguration{suffix}",
                    distributions=[distribution],
                    name=f"DistributionConfiguration{suffix}",
                )

                imagebuilder.CfnImagePipeline(
                    self,
                    f"{image_name}BuildPipeline",
                    infrastructure_configuration_arn=infrastructure_configuration.attr_arn,
                    name=image_name,
                    distribution_configuration_arn=distribution_configuration.attr_arn,
                    image_recipe_arn=image_recipe.attr_arn,
                )


class ServeStackSingleWorker(BaseStack):
    def __init__(
        self,
        scope,
        construct_id,
        *,
        instance_type,
        debug_mode=False,
    ):
        super().__init__(
            scope, construct_id, single_subnet=False, num_azs=1, debug_mode=debug_mode
        )

        launch_template = ec2.CfnLaunchTemplate(
            self,
            "LaunchTemplate",
            launch_template_data=ec2.CfnLaunchTemplate.LaunchTemplateDataProperty(
                # instance_market_options=ec2.CfnLaunchTemplate.InstanceMarketOptionsProperty(
                #     market_type="spot"
                # ),
                security_group_ids=[self.http_inbound_security_group.security_group_id],
            ),
        )

        worker = ec2.CfnInstance(
            self,
            "Worker",
            image_id=WHISPER_AMIS[get_ami_type(instance_type)],
            instance_type=instance_type,
            key_name=INTERNAL_SSH_KEY,
            launch_template=ec2.CfnInstance.LaunchTemplateSpecificationProperty(
                version="1", launch_template_id=launch_template.ref
            ),
            subnet_id=self.vpc.private_subnets[0].subnet_id,
            user_data=Fn.base64(get_user_data(InstanceRole.WORKER)),
        )

        ec2.CfnInstance(
            self,
            "Client",
            image_id=WHISPER_AMIS[get_ami_type(CLIENT_INSTANCE_TYPE)],
            instance_type=CLIENT_INSTANCE_TYPE,
            key_name=EXTERNAL_SSH_KEY,
            launch_template=ec2.CfnInstance.LaunchTemplateSpecificationProperty(
                version="1", launch_template_id=launch_template.ref
            ),
            subnet_id=self.vpc.public_subnets[0].subnet_id,
            user_data=Fn.base64(
                get_user_data(
                    InstanceRole.CLIENT, details={"endpoint": worker.attr_private_ip}
                )
            ),
        )


class ServeStack(BaseStack):
    def __init__(
        self,
        scope,
        construct_id,
        *,
        instance_type,
        with_autoscaling,
        debug_mode=False,
    ):
        num_azs = 2
        super().__init__(
            scope,
            construct_id,
            single_subnet=False,
            num_azs=num_azs,
            debug_mode=debug_mode,
        )

        target_group = elbv2.ApplicationTargetGroup(
            self,
            "TargetGroup",
            port=HTTP_PORT,
            protocol=elbv2.ApplicationProtocol.HTTP,
            vpc=self.vpc,
        )
        load_balancer = elbv2.ApplicationLoadBalancer(
            self,
            "LoadBalancer",
            idle_timeout=cdk.Duration.seconds(300),
            security_group=self.http_inbound_security_group,
            vpc=self.vpc,
            internet_facing=False,
        )
        listener = load_balancer.add_listener("Listener", port=HTTP_PORT, open=True)
        listener.add_target_groups("TargetGroup", target_groups=[target_group])

        if with_autoscaling:
            machine_image = ec2.MachineImage.generic_linux(
                {self.current_region: WHISPER_AMIS[get_ami_type(instance_type)]}
            )
            autoscaling_group = autoscaling.AutoScalingGroup(
                self,
                "AutoScalingGroup",
                vpc=self.vpc,
                min_capacity=1,
                max_capacity=4,
                instance_type=ec2.InstanceType(instance_type),
                machine_image=machine_image,
                security_group=self.http_inbound_security_group,
                user_data=ec2.UserData.custom(get_user_data(InstanceRole.WORKER)),
                key_name=INTERNAL_SSH_KEY,
                vpc_subnets=ec2.SubnetSelection(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
                ),
            )
            autoscaling_group.attach_to_application_target_group(target_group)
            autoscaling_group.scale_on_cpu_utilization(
                "ScalingPolicy", target_utilization_percent=30
            )
        else:
            workers = []

            for j in range(4):
                workers.append(
                    ec2.CfnInstance(
                        self,
                        f"Worker{j}",
                        image_id=WHISPER_AMIS[get_ami_type(instance_type)],
                        instance_type=instance_type,
                        key_name=INTERNAL_SSH_KEY,
                        subnet_id=self.vpc.private_subnets[j % num_azs].subnet_id,
                        security_group_ids=[
                            self.http_inbound_security_group.security_group_id
                        ],
                        user_data=Fn.base64(get_user_data(InstanceRole.WORKER)),
                    )
                )

            target_group.add_target(
                *[targets.InstanceIdTarget(worker.ref) for worker in workers]
            )

        ec2.CfnInstance(
            self,
            "Client",
            image_id=WHISPER_AMIS[get_ami_type(CLIENT_INSTANCE_TYPE)],
            instance_type=CLIENT_INSTANCE_TYPE,
            key_name=EXTERNAL_SSH_KEY,
            subnet_id=self.vpc.public_subnets[0].subnet_id,
            security_group_ids=[self.ssh_inbound_security_group.security_group_id],
            user_data=Fn.base64(
                get_user_data(
                    InstanceRole.CLIENT,
                    details={"endpoint": load_balancer.load_balancer_dns_name},
                )
            ),
        )


if __name__ == "__main__":
    app = App()

    BuildStack(app, "build-stack", volume_size=24, include_arm=False)

    for instance_type in [
        "t3.large",
        "m6a.large",
        "m6in.large",
        "c6a.xlarge",
        "g4dn.xlarge",
        "g5.xlarge",
    ]:
        if WHISPER_AMIS[get_ami_type(instance_type)] is not None:
            ServeStackSingleWorker(
                app,
                f"serve-stack-single-worker-{instance_type.replace('.', '-')}",
                instance_type=instance_type,
                debug_mode=True,
            )
        else:
            logging.info(f"AMI for instance type {instance_type} not available")

    instance_type = "t3.large"
    if WHISPER_AMIS[get_ami_type(instance_type)] is not None:
        for with_autoscaling in [True, False]:
            stack_name = f"serve-stack-{instance_type.replace('.', '-')}"

            if with_autoscaling:
                stack_name += "-autoscaling"

            ServeStack(
                app,
                stack_name,
                instance_type=instance_type,
                with_autoscaling=with_autoscaling,
                debug_mode=True,
            )
    else:
        logging.info(f"AMI for instance type {instance_type} not available")

    cloud_assembly = app.synth()
    save_templates(cloud_assembly)
