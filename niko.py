

import requests
from typing import List
from datetime import datetime
import time
import requests
from huggingface_hub import HfApi, upload_file, HfFolder, snapshot_download
from contextlib import contextmanager
import os
import sys
import shutil
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBOUNCE_TIME = 10  # seconds

class Push():
    def __init__(self):
        self.access_token = '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'
        HfFolder.save_token(self.access_token)
        self.api = HfApi()
        self.username = self.api.whoami(self.access_token)["name"]

        self.src_repo_url = 'locksA/5HBcDVUg1kC4qDha3vbVvhsy5foTFu6guCJ5vQ6p3N2Dv1Sh_vgg'
        self.hotkeys = [
            # # m
            # '5DLJiMEmqqsE1XPz9KvUvWaiBosW9EGHp8KRk48uoyhogvts',
            # '5EjbLoAZW4GB7wNPfhdRxvJrSKFDjbqWCQ1CLfk4brJnrJzG',
            # '5GuYucYYceR6Ypx9wAPzp7WmiRzBai4g9WPRW4UYvThX1XTA',
            # '5GuWwLASbBpDx1bfbtLh29VzW9otChTcyzKAySnMZrDuPGEU',
            # '5DACFwSSueVfoFYoJkVhPMjv2YepXnMM6cKgN3rXh8kVa1M3',
            # '5FERAEYhf7Wjo1QQ1pca9zqf1ZEHrAuJGzJi6Y5Sx2pZTxcG',
            # '5GsSNa8Xv3bywNtkzmqCuDsNMALTztXkpLTpnnQs3CPxMZnf',
            # '5CFCFBzJXZrekYxL1MedYwgQ6nhX5ycM6eTmVUTNBco3bGPp',
            # '5G3fAyAhi3DrSWycEPoRxMtBXdFuRdn2nV8R69eqRwq7u5Q8',
            # '5E7Xa1odn6n4ZQGykcgfZz4c2BT5xBVW8swLcM4DWUkwSb6c',
            
            
            '5CytxYe255hMSojev7kJASenxdFppriTnJMJ4RF3kVPg6v53',
            '5ERPG4ySpKZBWZACoZe7LnWGcZD5dGa2rAGSKQBMymXd8res',
            '5GvNMz73qNSQjFb9r4XPWbD747QH5jUAdRcxTfB1NwCJKN6m',
            '5HQpqmv5hjBr5NPsCqji9rMLNsFxLUeQaDg62x6pS65pTfCb',
            '5CV56xgcRU5XBmgQaPL2fhy6jsXHWfvgDvDxmjvDgpAxBAwh',
            '5GRKY95vuJKm1g6G9vzKRwNLoh1RdHoH5EnzhjakQD1afGnp',
            '5D5dcWavyswP6MghG8NDszoMfFJ8rwrhGPLNmqyWsutagr1Y',

            # # m1
            # '5GnEaJaYDbuM2RGCPNA26cbEu7o3Bj7PAeqpDQhVeRYWDNEz',
            # '5CaHf3YDPoJ2xxhdgk36wQYC9QwtaPzyPEpM8KRcTDEjTih4',
            # '5DvHAs3cF6dv3kqjncC5PR3FJ7xGpWeJUm5mfQN7UzCGWo92',
            # '5FpYewXnh4VSsNygiDGKq3YEE9AANPhWwHozVmgrywwLpGHa',
            # '5FS1wGYoP8ptn6hvYcX4tUzuHhwaZEtReUPTLM58UqYnEjGF',
            # '5EUvHph6VEEnqHcYGQEGiCj7EDmS3T1wEo4SkdWZ1B6uCx96',
            # '5EkTy46oWaUpseQqLpxS2TPBiF7YSrVkrzCPck9778b9vthD',
            # '5EyUP8HCeweMWiuqcRdobbmGeUEP6EkethP8Bs7rVV4dh4aG',
            # '5GWNaoJ5CfQtm9uPyLqEt3dcVN61Qekk3Th3ykf93XSbRUxm',
            # '5Ccp1DvrrXKQs9CAo9KHDiaV4x2QP14wpFt3LYD4yfm88PWa',

            
            # # # m2
            # '5FFBhMUAMXuQoHn52LU9X6oxafpZbwA7H7fsgKTS4cT1ifwp',
            # '5FZhkL8LrLE8AjGFV2tNbiSgE8DeJLhA6dNppzm6mcjY6Rw6',
            # '5CwKBpBByCnTA8DgsTsZjUibJkzbyLtUKPS56KduqeHXaDhr',
            # '5E4xEgZe7hat7MpSg8yooeLndW2hSduEDUb6QCj29qF5TwcP',
            # '5Fej6T2CUk5Vb3L2QzKWejqMMPpFZjQpffqkzYtdEzV4Xrfh',
            # '5GHP7eQ2bTHKiCPJ72NPWFNkyTCAetVFGBTR9s1b3W5YCgY1',
            # '5EenBfQcfXA4HjUcMPNDBxeFJePzAWd9zz6DJwxKKexxWuLq',
            # '5Cd7ou2Rw7D1TNwj9YJahTHiytA6ibmBWMnDjoT76ukEi5hi',
            # '5FW7jFTt1WdwVFYF7zpJfMpoZReACvqpCr5ynNYAMEMLfU5p',
            # '5D5PGDQAuxnsvn88N1Z8GvggPzcyLxFA8gJ17mZ4SBgttJHN',


            # # m3
            # '5DkpsYCiSbxUMgukbuzbjgq8g8WAW43k5XRbQ58VAjorZHF1',
            # '5C5eKB2eU3cepnMrM1k9s76SjzqxmSZhswAZ8FN4KdHfNxPT',
            # '5HasujtZDUcQjGjS94wTLGi5k4JdR63A6q8L44Yw6m1DTgvw',
            # '5Fe1CCph9tpJe1jsDLomqPs5hSuKbeH41Qf3enSdUNZBX7Rh',
            # '5Gc7Sde1fKaksuKjBmWBgT2JciKBC2ocLDokZjXDmuFBNPhz',
            # '5FW1o6WDb743ck7CQ3RF6UoeY2Yr63rmdQcjoBucStrKe1iW',
            # '5CFGm7hodyfkEmsh62FruLR5PS9UDFTaKVBMwQqaFeGuozjz',
            # '5HTXzgKNBNydzJcRq4B7PegFJQsi5iyh7pByiJbm8mS7bnRR',
            # '5GeKHjd5URy4srLNpHYE8sJEerTG5ossfQ7A2C9Q27UciQsY',
            # '5CyF4Kn3oyZnswMmb81mz4HUaiScA5xQaBW2RWHpbBxEiVMD',

            
            # # m4
            # '5EU5wHP3SURL2vjpT88LT2upDzD8QnvDYXLbjWicSWo9D8yK',
            # '5GYgq7jAvrBiUbmyZDvEmryZXiRaQxBnXGb6LZ3Bm23at6Gq',
            # '5EAcxpT5eBdp8FiigiapQbhK6WfnwyNnRidB11uojsqsBkLb',
            # '5GUJMGQPkN9GcXn29jG6efTkQCqLXDhfNdcFUcfgEYVuCQXH',
            # '5ENmuSrW7k119rocobGS6fANHeDLV7HppaZdVac3gfEgpCVh',
            # '5DZftZ6J9faPM5YSmvvBQGnbYjqkiVifVPTuYBgDxi8adxzy',
            # '5HMhRGeCkHPBKkr4XoguCjNLGKoq1XyTVu4qYF1bVx9jdU86',
            # '5GsxPLSqr2BXad7AnTntMTHJtbuDTV8KGsukvcfHQf6gT1Nu',
            # '5HN4PA8at1z4Qgwt6Xohjpw4qinMRvXMv72YACZR5MxSUggx',
            # '5DV5F4HW1yPJTvPqPqSaRenZiY8xQ8jtt582PjxbZapDaaUN',
        ]
        self.last_commit_time = time.time()

        self.local_dir = os.path.join(BASE_DIR, "healthcare/models/custom", self.src_repo_url)
        self.cache_dir = os.path.join(BASE_DIR, "healthcare/models/custom/cache")
        self.downloaded_num = 0

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
            

    def clean_cache(self):
        while True:
            try:
                shutil.rmtree(self.cache_dir)
                print(f'🧹 Cleaned the cache')
            except Exception as e:
                print(f'❌ exception while cleaning the cache, {e}')
            time.sleep(3600)

    def run(self):
        self.cleanThread = threading.Thread(target=self.clean_cache, daemon=True)
        self.cleanThread.start()
        while True:
            new_commit_time = self.commit_time(self.src_repo_url)
            if new_commit_time > self.last_commit_time:
                print(f"⌚ New commit detected at {new_commit_time} last_commit_time: {self.last_commit_time}")
                self.last_commit_time = new_commit_time
                time.sleep(DEBOUNCE_TIME)  # Debounce delay

                # Check for additional commits during the debounce period
                final_commit_time = self.commit_time(self.src_repo_url)
                if final_commit_time > self.last_commit_time:
                    print(f"⌚ Additional commit detected at {final_commit_time}")
                    self.last_commit_time = final_commit_time
                try:
                    if self.download_model():
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
                        
                    for hotkey in self.hotkeys:
                        dst_repo_url = self.username + "/" + hotkey + '_vgg'
                        print(f"⚠️ commit time of {hotkey} {self.commit_time(dst_repo_url)}")
                    print(f"💙 last commit time: {self.last_commit_time}")
                except Exception as e:
                    print(f"❌ Error occured while downloading and pushing : {e}")
                    return False
            time.sleep(1)

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    Push().run()