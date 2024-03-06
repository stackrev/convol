import bittensor as bt
import requests
from typing import List
from datetime import datetime
import time
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

src_repo_url = 'pmpmi/5FcFXiQamUrPb3g3cp3pHajuYDDUtDnPL3nYUFJn62f5Xx7G_vgg'

hotkeys = [
    '5FW7jFTt1WdwVFYF7zpJfMpoZReACvqpCr5ynNYAMEMLfU5p'
]

while True:
    try:
        if download_model(src_repo_url):
        bt.logging.success(last_commit_path)
    except Exception as e:
        bt.logging.error(f'Exception while downloading: {e}')