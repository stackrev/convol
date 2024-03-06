import bittensor as bt
import requests
from typing import List
from datetime import datetime
import time
import requests
from huggingface_hub import HfApi, upload_file, HfFolder, snapshot_download
from constants import BASE_DIR
from contextlib import contextmanager
import os
import sys
from dotenv import load_dotenv
load_dotenv()
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def download_model(src_repo_url: str):
    access_token = os.getenv('ACCESS_TOKEN')
    HfFolder.save_token(access_token)
    # Download the model
    try:
        local_dir = os.path.join(BASE_DIR, "healthcare/models/custom", src_repo_url)
        cache_dir = os.path.join(BASE_DIR, "healthcare/models/custom/cache")
        with suppress_stdout_stderr():
            snapshot_download(repo_id = src_repo_url, local_dir = local_dir, token = os.getenv('ACCESS_TOKEN'), cache_dir = cache_dir)
        bt.logging.info(f"✅ Successfully downloaded the model {src_repo_url}.")
        return True
    except Exception as e:
        bt.logging.error(f"❌ Error occured while downloading the model of miner {uid} : {e}")
        return False            

def commit_time(model_path: str):
    """
    This method returns the commit time of the model.

    Args:
    - model_path (str): The path of model.

    Returns:
    - int: The last commit time of the models in the form of integer.
    """
    try:
        api_url = f"https://huggingface.co/api/models/{model_path}"
        response = requests.get(api_url)

        if response.status_code == 200:
            model_info = response.json()
            commit_time = model_info.get("lastModified")
            
            # Parse the date string to a datetime object
            date_obj = datetime.fromisoformat(commit_time[:-1])  # Removing 'Z' at the end

            # Convert the datetime object to UNIX timestamp (integer)
            timestamp = int(date_obj.timestamp())

            return timestamp
        else:
            return float('inf')
    except Exception as e:
        return float('inf')

last_commit_path = 'wh3/5GZaXpEyDd8RSHsfsbwBqRevNJtsyQ5T8x45jr4LX5L7v3s3_vgg'
download_time = 0
while True:
    try:
        if commit_time(last_commit_path) > download_time:
            if download_model(last_commit_path):
                with open('healthcare/models/custom/current_model_repo', 'w') as f:
                    f.write(last_commit_path)
                    download_time = time.time()
                    bt.logging.success(last_commit_path)
        time.sleep(10)
    except Exception as e:
        bt.logging.error(f'Exception: {e}')