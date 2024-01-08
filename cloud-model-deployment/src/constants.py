HTTP_PORT = 8080
SSH_PORT = 22

DEFAULT_REGION = "eu-central-1"

EC2_USER = "ec2-user"
EC2_HOME_FOLDER = f"/home/{EC2_USER}"
REPO_NAME = "public-repo"
FOLDER_NAME = "cloud-model-deployment"

BUILD_INSTANCE_TYPES = {
    ("x86", "cpu"): "t3.large",
    ("x86", "gpu"): "g4dn.xlarge",
    ("arm", "cpu"): "t4g.large",
    ("arm", "gpu"): "g5g.xlarge",
}
CLIENT_INSTANCE_TYPE = "t3.medium"
