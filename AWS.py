#https://aws.amazon.com/tutorials/machine-learning-tutorial-deploy-model-to-real-time-inference-endpoint/
import pandas as pd
import numpy as np
import boto3
import sagemaker
import time
import json
import io
from io import StringIO
import base64
import pprint
import re

from sagemaker.image_uris import retrieve

sess = sagemaker.Session()
write_bucket = sess.default_bucket()
write_prefix = "fraud-detect-demo"

region = sess.boto_region_name
s3_client = boto3.client("s3", region_name=region)
sm_client = boto3.client("sagemaker", region_name=region)
sm_runtime_client = boto3.client("sagemaker-runtime")
sm_autoscaling_client = boto3.client("application-autoscaling")

sagemaker_role = sagemaker.get_execution_role()


# S3 locations used for parameterizing the notebook run
read_bucket = "sagemaker-sample-files"
read_prefix = "datasets/tabular/synthetic_automobile_claims"
model_prefix = "models/xgb-fraud"

data_capture_key = f"{write_prefix}/data-capture"

# S3 location of trained model artifact
model_uri = f"s3://{read_bucket}/{model_prefix}/fraud-det-xgb-model.tar.gz"

# S3 path where data captured at endpoint will be stored
data_capture_uri = f"s3://{write_bucket}/{data_capture_key}"

# S3 location of test data
test_data_uri = f"s3://{read_bucket}/{read_prefix}/test.csv"




# Create a Real-Time Inference endpoint

# Retrieve the SageMaker managed XGBoost image
training_image = retrieve(framework="xgboost", region=region, version="1.3-1")

# Specify a unique model name that does not exist
model_name = "fraud-detect-xgb"
primary_container = {
                     "Image": training_image,
                     "ModelDataUrl": model_uri
                    }

model_matches = sm_client.list_models(NameContains=model_name)["Models"]
if not model_matches:
    model = sm_client.create_model(ModelName=model_name,
                                   PrimaryContainer=primary_container,
                                   ExecutionRoleArn=sagemaker_role)
else:
    print(f"Model with name {model_name} already exists! Change model name to create new")




# Endpoint Config name
endpoint_config_name = f"{model_name}-endpoint-config"

# Endpoint config parameters
production_variant_dict = {
                           "VariantName": "Alltraffic",
                           "ModelName": model_name,
                           "InitialInstanceCount": 1,
                           "InstanceType": "ml.m5.xlarge",
                           "InitialVariantWeight": 1
                          }

# Data capture config parameters
data_capture_config_dict = {
                            "EnableCapture": True,
                            "InitialSamplingPercentage": 100,
                            "DestinationS3Uri": data_capture_uri,
                            "CaptureOptions": [{"CaptureMode" : "Input"}, {"CaptureMode" : "Output"}]
                           }


# Create endpoint config if one with the same name does not exist
endpoint_config_matches = sm_client.list_endpoint_configs(NameContains=endpoint_config_name)["EndpointConfigs"]
if not endpoint_config_matches:
    endpoint_config_response = sm_client.create_endpoint_config(
                                                                EndpointConfigName=endpoint_config_name,
                                                                ProductionVariants=[production_variant_dict],
                                                                DataCaptureConfig=data_capture_config_dict
                                                               )
else:
	print(f"Endpoint config with name {endpoint_config_name} already exists! Change endpoint config name to create new")




endpoint_name = f"{model_name}-endpoint"

endpoint_matches = sm_client.list_endpoints(NameContains=endpoint_name)["Endpoints"]
if not endpoint_matches:
    endpoint_response = sm_client.create_endpoint(
                                                  EndpointName=endpoint_name,
                                                  EndpointConfigName=endpoint_config_name
                                                 )
else:
    print(f"Endpoint with name {endpoint_name} already exists! Change endpoint name to create new")

resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
status = resp["EndpointStatus"]
while status == "Creating":
    print(f"Endpoint Status: {status}...")
    time.sleep(60)
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
print(f"Endpoint Status: {status}")


resp = sm_client.describe_endpoint(EndpointName=endpoint_name)

# SageMaker expects resource id to be provided with the following structure
resource_id = f"endpoint/{endpoint_name}/variant/{resp['ProductionVariants'][0]['VariantName']}"

# Scaling configuration
scaling_config_response = sm_autoscaling_client.register_scalable_target(
                                                          ServiceNamespace="sagemaker",
                                                          ResourceId=resource_id,
                                                          ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                                                          MinCapacity=1,
                                                          MaxCapacity=2
                                                        )