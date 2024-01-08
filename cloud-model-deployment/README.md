## Introduction

This repo shows how to deploy Whisper on AWS using EC2 instances (without using Docker). Whisper is a state-of-the-art speech-to-text model developed by OpenAI (https://openai.com/research/whisper).

**Note**: I am assuming that you have AWS CLI installed and that you are authenticating yourself as a user that has the right credentials to create resource stacks, start EC2 instances, etc.

**Warning**: performing the following steps will incur charges to your AWS account. To minimise the cost, please perform the clean-up step once you're done.

## Deploy Whisper

We will call the final deployment a **serve stack** and it will consist of 2 components:
1. an endpoint capable of receiving and processing requests (this is either a single EC2 worker instance or a load balancer pointing at a group of workers),
2. a client EC2 instance from which we can submit requests.

By logging into the client we will be able to test the latency and throughput of the endpoint. To access the client you must generate a key pair in your AWS account and pass it when you create the EC2 instance. We will refer to this key pair as the `EXTERNAL_SSH_KEY`.

For debugging purposes it might be convenient to have another key pair that will be used by the client to access the workers (if necessary). We will refer to this key pair as the `INTERNAL_SSH_KEY`.. Finally, if you also store this key in `cloud-model-deployment/keys/internal_ssh_key.pem`, then it will be automatically available on the client.

To deploy the model follow these steps:

1. Clone the repo and go to the `cloud-model-deployment` folder.
2. Make sure you have a Python `venv` with `aws-cdk-lib` installed (can be installed via `pip`).
3. In `src/config.py` set `EXTERNAL_SSH_KEY` and optionally `INTERNAL_SSH_KEY`.
4. Run `python -m src.generate_infra` to generate a **build stack** template in the `cfn_templates` subfolder. Build stack only contains recipes for building images to be used in the serve stack.
5. Deploy the build stack by calling
```
aws cloudformation deploy --template-file cfn_templates/build-stack.json --stack-name build-stack --capabilities CAPABILITY_NAMED_IAM
```
5. Once the build stack has been deployed, you should go to the AWS EC2 ImageBuilder (https://eu-central-1.console.aws.amazon.com/imagebuilder/home) and click on Image pipelines. There, you should find two image pipelines: `WhisperCPUX86` and `WhisperGPUX86`. The first one is for running inference on CPU, while the second allows us to use GPU (it has CUDA installed). 
6. Run the pipelines to build an images. In my case building the CPU image took 20 mins, while the GPU image took 32 mins. Once the images are built, you should find their AMI ids in EC2 -> Images -> AMIs and paste them into `src/config.py`.
7. Once we have the AMIs, we can run `python -m src.generate_infra` again and this time it will generate templates for the serve stack. The only difference is in the endpoint:
   - `serve-stack-single-worker-{INSTANCE_TYPE}`: the endpoint consists of a single EC2 of specified instance type
   - `serve-stack-t3-large`: the endpoint consists of 4 `t3.large` instances behind a load balancer
   - `serve-stack-t3-large-autoscaling`: the endpoint consists of an autoscaling group of `t3.large` instances behind a load balancer
8. To deploy the simplest serve stack call
```
aws cloudformation deploy --template-file cfn_templates/serve-stack-single-worker-t3-large.json --stack-name serve-stack-1 --capabilities CAPABILITY_NAMED_IAM
```
9. Once the stack has been deployed, log in to the client (you will find its IP address in the AWS EC2 section) and run:
```
python -m src.client
```

## Performance assessment

The serve stacks can be used to assess the relative performance of various instance types provided by AWS. In fact, that is precisely what `src/client.py` does: it sends multiple asynchronous requests to the endpoint (each request corresponds to transcribing a short audio) and measures the time taken by the worker. A reasonable figure of merit for this task is the real-time factor defined as the ratio of the length of the transcribed audio file to the processing time. E.g. a real-time factor of 3 means that within 1 second our endpoint transcribes 3 seconds of audio. 

Here are the preliminary results I have obtained for the `base` and `medium` models (I have excluded the stack that involves autoscaling, as it would not be a fair comparison).

|                            | `base` | `medium` |
|----------------------------|--------|----------|
| `t3.large`                 | 1.4    | 0.1      |
| `m6a.large`                | 2.9    | 0.3      |
| `c6a.xlarge`               | 4.3    | 0.5      |
| `g4dn.xlarge`              | 7.1    | 0.8      |
| `g5.xlarge`                | 11.2   | 1.3      |
| 4 instances of `t3.large`  | 4.1    | 0.3      |

So we managed to get the models running, but there are several aspects that could be optimised:
- at the moment we are using `torch` in the so-called eager mode, one could experiment whether using `torch.compile` with various backends leads to improved performance
- if we look at the CPU instances, it is clear that we are not utilising all the cores efficiently
- if we look at the GPU instances, it is clear that we do not get particularly high GPU usage

The low CPU/GPU usage is probably due to the fact that we are processing relatively small batches of data. This might be similar to what happens if we use Whisper for real-time transcription (but note that the original implementation of Whisper is not really streaming-friendly).

## Clean up

1. Delete all the stacks in CloudFormation.
2. Delete all the AMIs you have created and the associated snapshots.
