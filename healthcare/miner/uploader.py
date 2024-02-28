from huggingface_hub import HfApi, upload_file, HfFolder, snapshot_download
from constants import BASE_DIR
import bittensor as bt
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
def upload_model(hotkey, src_repo_url):
    access_token = os.getenv('ACCESS_TOKEN')
    HfFolder.save_token(access_token)
    try:
        api = HfApi()
        username = api.whoami(access_token)["name"]
        dst_repo_url = username + "/" + hotkey + '_cnn'
        api.create_repo(token=access_token, repo_id=dst_repo_url, exist_ok = True)

        # Download the model
        try:
            local_dir = os.path.join(BASE_DIR, "healthcare/models/custom", src_repo_url)
            cache_dir = os.path.join(BASE_DIR, "healthcare/models/custom/cache")
            with suppress_stdout_stderr():
                snapshot_download(repo_id = src_repo_url, local_dir = local_dir, token = os.getenv('ACCESS_TOKEN'), cache_dir = cache_dir)
            bt.logging.info(f"✅ Successfully downloaded the given model.")

            # Upload it to the huggingface
            try:
                for root, dirs, files in os.walk(local_dir):
                    for file in files:
                        # Generate the full path and then remove the base directory part
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, local_dir)
                        with suppress_stdout_stderr():
                            upload_file(
                                path_or_fileobj=full_path,
                                path_in_repo=relative_path,
                                repo_id=dst_repo_url
                            )

                bt.logging.info(f"✅ Best model uploaded at {dst_repo_url}")
            except Exception as e:
                bt.logging.error(f"❌ Error occured while pushing recent model to a repository : {e}")
        except Exception as e:
            bt.logging.error(f"❌ Error occured while downloading the model of miner {uid} : {e}")
            return ""

    except Exception as e:
        bt.logging.error(f"❌ Error occured while creating a repository : {e}")