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

DEBOUNCE_TIME = 10  # seconds

class Push():
    def __init__(self):
        self.access_token = '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'
        HfFolder.save_token(self.access_token)
        self.api = HfApi()
        self.username = self.api.whoami(self.access_token)["name"]

        self.src_repo_url = 'wh3/5GZaXpEyDd8RSHsfsbwBqRevNJtsyQ5T8x45jr4LX5L7v3s3_vgg'
        self.hotkeys = [
            '5CyF4Kn3oyZnswMmb81mz4HUaiScA5xQaBW2RWHpbBxEiVMD',
        ]
        self.last_commit_time = time.time()

        self.local_dir = os.path.join(BASE_DIR, "healthcare/models/custom", self.src_repo_url)
        self.cache_dir = os.path.join(BASE_DIR, "healthcare/models/custom/cache")

        # Create asyncio event loop to manage async tasks.
        self.threads = []


    def download_model(self):
        # Download the model
        try:
            with suppress_stdout_stderr():
                snapshot_download(repo_id = self.src_repo_url, local_dir = self.local_dir, token = self.access_token, cache_dir = self.cache_dir)
            return True
        except Exception as e:
            print(f"❌ Error occured while downloading the model {self.src_repo_url} : {e}")
            return False          

    def commit_time(self, model_path: str):
        try:
            api_url = f"https://huggingface.co/api/models/{model_path}"
            response = requests.get(api_url)

            if response.status_code == 200:
                model_info = response.json()
                lastModified = model_info.get("lastModified")
                
                # Parse the date string to a datetime object
                date_obj = datetime.fromisoformat(lastModified[:-1])  # Removing 'Z' at the end

                # Convert the datetime object to UNIX timestamp (integer)
                return date_obj.timestamp()
            else:
                return float('inf')
        except Exception as e:
            return float('inf')


    def push_model(self, hotkey: str):
        try:
            dst_repo_url = self.username + "/" + hotkey + '_vgg'
            self.api.create_repo(token=self.access_token, repo_id=dst_repo_url, exist_ok = True)

            # Upload it to the huggingface
            try:
                print(f'🎲 try pushing to {hotkey} at {time.time()}')
                for root, dirs, files in os.walk(self.local_dir):
                    for file in files:
                        # Generate the full path and then remove the base directory part
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, self.local_dir)
                        with suppress_stdout_stderr():
                            upload_file(
                                path_or_fileobj=full_path,
                                path_in_repo=relative_path,
                                repo_id=dst_repo_url
                            )
                print(f"✅ Model uploaded to {hotkey} at {time.time()}")
                return True

            except Exception as e:
                print(f"❌ Error occured while pushing the model to a repository : {e}")
                return False
            
        except Exception as e:
            print(f"❌ Error occured while creating a repository : {e}")
            return False

    def run(self):
        try:
            while True:
                downloaded = self.download_model()
                if downloaded:
                    print(f"👍 downloaded the model at {time.time()}")
                    self.threads = []
                    for hotkey in self.hotkeys:
                        thread = threading.Thread(target=self.push_model, args=(hotkey,))
                        thread.start()
                        self.threads.append(thread)
                    # Wait for all threads to complete
                    for thread in self.threads:
                        thread.join()
                    print('all threads done')
                    break
        
            for hotkey in self.hotkeys:
                dst_repo_url = self.username + "/" + hotkey + '_vgg'
                print(f"⚠️ commit time of {hotkey} {self.commit_time(dst_repo_url)}")
            print(f"💙 last commit time: {self.last_commit_time}")
        except Exception as e:
            print(f"❌ Error occured while downloading and pushing : {e}")
            return False

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    Push().run()