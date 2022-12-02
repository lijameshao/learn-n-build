"""
Use case:
    When you drag and drop local files onto S3, some files may fail to upload
    Instead of having to check which files has been uploaded and which not
    List files in the S3 folder, compare to local, copy files that have not
    been uploaded to a different folder, such that you can drag and drop
    the missing files
"""

import boto3
import os
import shutil

AWS_PROFILE = "YOUR_AWS_PROFILE"
bucket = "YOUR_BUCKET"
prefix = "YOUR_FOLDER_PREFIX/"
local_folder = "YOUR_LOCAL_FOLDER"
local_folder_to_upload = "YOUR_LOCAL_FOLDER_TO_UPLOAD"

session = boto3.session.Session(profile_name=AWS_PROFILE)
s3_client = session.client("s3")

response = s3_client.list_objects_v2(Bucket=bucket,Prefix=prefix)

results = []
for content in response.get("Contents", []):
    results.append(content["Key"].split("/")[-1])

local_files = os.listdir(local_folder)
to_upload = list(set(local_files) - set(results))

for fname in to_upload:
    src = f"{local_folder}/{fname}"
    dst = f"{local_folder_to_upload}/{fname}"
    shutil.copyfile(src, dst)
