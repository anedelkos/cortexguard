{
  "Version": "2020-12-01",
  "Metadata": {},
  "Steps": [
    {
      "Name": "RetrainStepClassifier",
      "Type": "Processing",
      "Arguments": {
        "ProcessingResources": {
          "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.t3.medium",
            "VolumeSizeInGB": 30
          }
        },
        "AppSpecification": {
          "ImageUri": "${sagemaker_sklearn_image}",
          "ContainerEntrypoint": [
            "python",
            "/opt/ml/processing/code/run.py"
          ]
        },
        "RoleArn": "${sagemaker_role_arn}",
        "NetworkConfig": {
          "EnableNetworkIsolation": false,
          "VpcConfig": {
            "SecurityGroupIds": ${sagemaker_sg_ids},
            "Subnets": ${subnet_ids}
          }
        },
        "ProcessingInputs": [
          {
            "InputName": "code",
            "AppManaged": false,
            "S3Input": {
              "S3Uri": "s3://${code_bucket}/code",
              "LocalPath": "/opt/ml/processing/code",
              "S3DataType": "S3Prefix",
              "S3InputMode": "File",
              "S3DataDistributionType": "FullyReplicated"
            }
          }
        ],
        "ProcessingOutputConfig": {
          "Outputs": [
            {
              "OutputName": "training_result",
              "S3Output": {
                "S3Uri": "s3://${output_bucket}/artifacts",
                "LocalPath": "/opt/ml/processing/output",
                "S3UploadMode": "EndOfJob"
              }
            }
          ]
        },
        "Environment": {
          "DB_URL_SECRET_ARN": "${secret_arn}"
        }
      },
      "DependsOn": []
    },
    {
      "Name": "RegisterStepClassifier",
      "Type": "RegisterModel",
      "Arguments": {
        "ModelPackageGroupName": "${model_package_group_name}",
        "ModelMetrics": {
          "ModelQuality": {
            "Statistics": {
              "ContentType": "application/json",
              "S3Uri": "s3://${output_bucket}/artifacts/training_result.json"
            }
          }
        },
        "InferenceSpecification": {
          "Containers": [
            {
              "Image": "${sagemaker_sklearn_image}",
              "ModelDataUrl": "s3://${output_bucket}/artifacts/model.tar.gz"
            }
          ],
          "SupportedContentTypes": [
            "text/csv",
            "application/json"
          ],
          "SupportedResponseMIMETypes": [
            "application/json"
          ]
        }
      },
      "DependsOn": [
        "RetrainStepClassifier"
      ]
    },
    {
      "Name": "ComputeMonitorBaseline",
      "Type": "Processing",
      "Arguments": {
        "ProcessingResources": {
          "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.t3.medium",
            "VolumeSizeInGB": 20
          }
        },
        "AppSpecification": {
          "ImageUri": "${sagemaker_model_monitor_image}"
        },
        "RoleArn": "${sagemaker_role_arn}",
        "ProcessingInputs": [
          {
            "InputName": "baseline_dataset",
            "AppManaged": false,
            "S3Input": {
              "S3Uri": "s3://${output_bucket}/data-capture",
              "LocalPath": "/opt/ml/processing/baseline",
              "S3DataType": "S3Prefix",
              "S3InputMode": "File",
              "S3DataDistributionType": "FullyReplicated"
            }
          }
        ],
        "ProcessingOutputConfig": {
          "Outputs": [
            {
              "OutputName": "baseline_output",
              "S3Output": {
                "S3Uri": "s3://${output_bucket}/monitor-baseline",
                "LocalPath": "/opt/ml/processing/output",
                "S3UploadMode": "EndOfJob"
              }
            }
          ]
        }
      },
      "DependsOn": [
        "RegisterStepClassifier"
      ]
    }
  ]
}
