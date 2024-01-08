## Introduction

This repo shows how to deploy Whisper on AWS using EC2 instances (without using Docker). Whisper is a state-of-the-art speech-to-text model developed by OpenAI (https://openai.com/research/whisper).

**Note**: I am assuming that you have AWS CLI installed and that you are authenticating yourself as a user that has the right credentials to create resource stacks, start EC2 instances, etc.

**Warning**: performing the following steps will incur charges to your AWS account. To minimise the cost, please perform the clean-up step once you're done.

## Deploy Whisper

The final deployment will consist of 2 components:
1. an endpoint capable of receiving and processing requests (this is either a single EC2 worker instance or a load balancer pointing at a group of workers),
2. a client EC2 instance from which we can submit requests.

By logging into the client we will be able to test the latency and throughput of the endpoint. To access the client you must generate a key pair in your AWS account and pass it when you create the EC2 instance. The name of the key used to access the client must be specified in the `EXTERNAL_SSH_KEY` environmental variable.

For debugging purposes it might be convenient to have another key pair that will be used by the client to access the workers (if necessary). To specify this key pair use the `INTERNAL_SSH_KEY` environmental variable. Finally, if you also store this key in `cloud-model-deployment/keys/internal_ssh_key.pem`, then it will be automatically available on the client.

To deploy the model follow these steps:

1. Clone the repo and go to the `cloud-model-deployment` folder.
2. Make sure you have a Python `venv` with `aws-cdk-lib` installed (can be installed via `pip`).
3. Set environmental variables specifying SSH keys, e.g.:
```
export EXTERNAL_SSH_KEY=external_ssh_key
export INTERNAL_SSH_KEY=internal_ssh_key
```
4. Run `python -m src.generate_infra` to generate a build stack template in the `cfn_templates` subfolder.
5. Deploy the build stack by calling
```
aws cloudformation deploy --template-file cfn_templates/build-stack.json --stack-name build-stack --capabilities CAPABILITY_NAMED_IAM
```
5. Once the build stack has been deployed, you should go to the AWS EC2 ImageBuilder (https://eu-central-1.console.aws.amazon.com/imagebuilder/home) and click on Image pipelines. There, you should find two image pipelines: `WhisperCPUX86` and `WhisperGPUX86`. If you run such a pipeline it will build an image that we can later use for our EC2 instances. The `GPU`



## Clean up

1. Delete all the stacks in CloudFormation.
2. Delete all the AMIs you have created and the associated snapshots.
