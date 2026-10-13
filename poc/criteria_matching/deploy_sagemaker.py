"""
Deploy / delete the matcher model as a SageMaker real-time endpoint (vLLM).

The endpoint serves the model with vLLM through an AWS-provided container
and accepts the OpenAI chat completions payload, which is what
backends.SageMakerBackend sends. The same matcher.py code runs locally and
against the endpoint.

Requirements: AWS credentials, an IAM execution role for SageMaker, quota for
the instance type, and `uv sync --extra sagemaker`.

Container: pass --image-uri explicitly. Use an LMI (djl-inference) image whose
vLLM is >= 0.8.5 (first version with Qwen3). Find the current tags in
https://github.com/aws/deep-learning-containers/blob/master/available_images.md
(e.g. 763104351884.dkr.ecr.<region>.amazonaws.com/djl-inference:<tag>-lmi<N>...).

Usage:
    uv run --extra sagemaker poc/criteria_matching/deploy_sagemaker.py deploy \
        --name qwen3-4b-matcher --role arn:aws:iam::<account>:role/<sagemaker-role> --image-uri <lmi image>
    uv run --extra sagemaker poc/criteria_matching/deploy_sagemaker.py delete --name qwen3-4b-matcher

Cost: the endpoint bills per hour while it exists (~US$1-1.5/h for ml.g5.xlarge
or ml.g6.xlarge, depending on region). Delete it when you are done testing.
"""

import argparse

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# 24 GB GPU (A10G / L4): the 4B model in bf16 (~8 GB) plus room for the KV cache.
DEFAULT_INSTANCE = "ml.g6.xlarge"


def deploy(sm, args):
    env = {
        "HF_MODEL_ID": args.model,
        "OPTION_ASYNC_MODE": "true",
        "OPTION_ROLLING_BATCH": "disable",
        "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
        "OPTION_MAX_MODEL_LEN": str(args.max_model_len),
        "OPTION_TENSOR_PARALLEL_DEGREE": "1",
    }
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
    sm.create_model(ModelName=args.name, ExecutionRoleArn=args.role,
                    PrimaryContainer={"Image": args.image_uri, "Environment": env})
    sm.create_endpoint_config(EndpointConfigName=args.name, ProductionVariants=[{
        "VariantName": "main", "ModelName": args.name, "InstanceType": args.instance_type,
        "InitialInstanceCount": 1, "ContainerStartupHealthCheckTimeoutInSeconds": 900,
    }])
    sm.create_endpoint(EndpointName=args.name, EndpointConfigName=args.name)
    print(f"Creating endpoint {args.name} on {args.instance_type} (takes ~10 min)...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=args.name, WaiterConfig={"Delay": 30, "MaxAttempts": 60})
    print(f"Endpoint {args.name} is InService. Remember to delete it when done (it bills per hour).")


def delete(sm, args):
    for call, kwargs in ((sm.delete_endpoint, {"EndpointName": args.name}),
                         (sm.delete_endpoint_config, {"EndpointConfigName": args.name}),
                         (sm.delete_model, {"ModelName": args.name})):
        try:
            call(**kwargs)
            print(f"{call.__name__}: {args.name}")
        except sm.exceptions.ClientError as e:
            print(f"{call.__name__}: {e.response['Error']['Message']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy/delete the matcher SageMaker endpoint")
    parser.add_argument("action", choices=["deploy", "delete"])
    parser.add_argument("--name", default="qwen3-4b-matcher", help="Name used for model, endpoint config and endpoint")
    parser.add_argument("--region")
    parser.add_argument("--role", help="[deploy] SageMaker execution role ARN")
    parser.add_argument("--image-uri", help="[deploy] LMI container image (vLLM >= 0.8.5)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="[deploy] HF model id")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE)
    parser.add_argument("--max-model-len", type=int, default=8192, help="[deploy] max tokens per request (prompt + answer)")
    parser.add_argument("--hf-token", help="[deploy] only for gated/private models")
    args = parser.parse_args()

    import boto3

    sm = boto3.client("sagemaker", region_name=args.region)
    if args.action == "deploy":
        if not (args.role and args.image_uri):
            parser.error("deploy needs --role and --image-uri")
        deploy(sm, args)
    else:
        delete(sm, args)
