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
import threading
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

access_token = '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'
HfFolder.save_token(access_token)
api = HfApi()
username = api.whoami(access_token)["name"]

src_repo_url = 'wh3/5GZaXpEyDd8RSHsfsbwBqRevNJtsyQ5T8x45jr4LX5L7v3s3_vgg'
hotkeys = [
    '5GYKohGNwBmZyHsxNV2dedmgHgnjALgoRZb1Cs8uGhEHcrcH',
    '5CzHzrs18jNMPxESRNT34BaYDzFymHd7KmutfzWcu1uGRNV3',
    '5Fnpq3pQE2D7Lm1RkvdteAhWvrEb8YGRJaKCsJdgnHpUdVBr',
]

local_dir = os.path.join(BASE_DIR, "healthcare/models/custom", src_repo_url)
cache_dir = os.path.join(BASE_DIR, "healthcare/models/custom/cache")

DEBOUNCE_TIME = 10  # seconds

def download_model(src_repo_url: str):
    # Download the model
    try:
        with suppress_stdout_stderr():
            snapshot_download(repo_id = src_repo_url, local_dir = local_dir, token = access_token, cache_dir = cache_dir)
        return True
    except Exception as e:
        print(f"❌ Error occured while downloading the model {src_repo_url} : {e}")
        return False          

def commit_time(model_path: str):
    try:
        api_url = f"https://huggingface.co/api/models/{model_path}"
        response = requests.get(api_url)

        if response.status_code == 200:
            model_info = response.json()
            commit_time = model_info.get("lastModified")
            
            # Parse the date string to a datetime object
            date_obj = datetime.fromisoformat(commit_time[:-1])  # Removing 'Z' at the end

            # Convert the datetime object to UNIX timestamp (integer)
            return date_obj.timestamp()
        else:
            return float('inf')
    except Exception as e:
        return float('inf')


def push_model(hotkey: str):
    try:
        dst_repo_url = username + "/" + hotkey + '_vgg'
        api.create_repo(token=access_token, repo_id=dst_repo_url, exist_ok = True)

        # Upload it to the huggingface
        try:
            print(f'🎲 try pushing to {hotkey} at {time.time()}')
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
            print(f"✅ Model uploaded to {hotkey} at {time.time()}")
        except Exception as e:
            print(f"❌ Error occured while pushing the model to a repository : {e}")

    except Exception as e:
        print(f"❌ Error occured while creating a repository : {e}")


def main():
    last_commit_time = 0
    while True:
        try:
            new_commit_time = commit_time(src_repo_url)
            if new_commit_time > last_commit_time:
                print(f"⌚ New commit detected at {new_commit_time} last_commit_time: {last_commit_time}")
                last_commit_time = new_commit_time
                time.sleep(DEBOUNCE_TIME)  # Debounce delay

                # Check for additional commits during the debounce period
                final_commit_time = commit_time(src_repo_url)
                if final_commit_time > last_commit_time:
                    print(f"⌚ Additional commit detected at {final_commit_time}")
                    last_commit_time = final_commit_time
                if download_model(src_repo_url):
                    print(f"👍 downloaded the model at {time.time()}")
                    threads = []
                    for hotkey in hotkeys:
                        thread = threading.Thread(target=push_model, args=(hotkey,))
                        thread.start()
                        threads.append(thread)
                    for thread in threads:
                        thread.join()
            
                for hotkey in hotkeys:
                    dst_repo_url = username + "/" + hotkey + '_vgg'
                    print(f"⚠️ commit time of {hotkey} {commit_time(dst_repo_url)}")
                print(f"💙 last commit time: {last_commit_time}")
            
            time.sleep(30)

        except Exception as e:
            print(f'❌Exception while running: {e}')
        time.sleep(1)

main()