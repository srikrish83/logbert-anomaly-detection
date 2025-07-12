import boto3
from src.utils.config_loader import load_config

config = load_config()

s3 = boto3.client(
    's3',
    aws_access_key_id=config['aws']['access_key'],
    aws_secret_access_key=config['aws']['secret_key'],
    region_name=config['aws']['region']
)

def upload_to_s3(file_path, s3_key):
    s3.upload_file(file_path, config['aws']['bucket'], s3_key)
    print(f"✅ Uploaded {file_path} to S3 as {s3_key}")

def download_from_s3(s3_key, file_path):
    s3.download_file(config['aws']['bucket'], s3_key, file_path)
    print(f"✅ Downloaded {s3_key} from S3 to {file_path}")
