# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 demon

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import random
import shutil
import sys
from datetime import datetime
from contextlib import contextmanager
import torch
import numpy as np
import bittensor as bt
import requests
from typing import List
from tensorflow.keras.models import load_model
from healthcare.dataset.dataset import load_dataset, load_and_preprocess_image
from constants import BASE_DIR


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

def get_loss(
    model_paths: List[str],
    uids: torch.LongTensor
) -> List[float]:
    """
    This method returns a loss value for the model, which is used to update the miner's score.

    Args:
    - model_paths (List[str]): The path of models.
    - uids (torch.LongTensor): The uid of models.

    Returns:
    - List[int, float]: The loss value for the models.
    """
    try:
        # Load dataset
        csv_path = os.path.join(BASE_DIR, 'healthcare/dataset/validator/Data_Entry.csv')
        image_dir = os.path.join(BASE_DIR, 'healthcare/dataset/validator/images')
        image_paths, binary_output, dataframe = load_dataset(csv_path, image_dir)

        # Generate x_input and y_output
        x_input = []
        y_output = []
        for idx, image_path in enumerate(image_paths):
            img = load_and_preprocess_image(os.path.join(image_dir, image_path))
            if isinstance(img, str):
                continue
            x_input.append(img)
            y_output.append(binary_output[idx])
        bt.logging.info(f"✅ Successfully loaded dataset.")
    except Exception as e:
        bt.logging.error(f"❌ Error occured while loading dataset : {e}")
        return []

    # Load model
    loss_of_models = []
    for idx, model_path in enumerate(model_paths):
        # Check if model exists
        if not model_path:
            loss = float('inf')
        else:
            bt.logging.info(f"⚒️  Processing the model of miner {uids[idx]} ...")
            try:
                model = load_model(model_path)
                # Evaluate loss and accuracy
                with suppress_stdout_stderr():
                    loss, accuracy = model.evaluate(np.array(x_input), np.array(y_output), verbose=0)
            except Exception as e:
                # bt.logging.error(f"❌ Error occured while loading model : {e}")
                loss = float('inf')
        loss_of_models.append([idx, loss])
    return loss_of_models

def get_rewards(
    self,
    model_paths: List[str],
    uids: torch.LongTensor,
    ips: List[str],
    commit_blocks: List[int],
    repo_ids: List[str]
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given models.

    Args:
    - model_paths (List[str]): A list of path to models.
    - uids (torch.LongTensor): A list of uids.
    - ips (List[str]): A list of ip addresses.
    - commit_blocks (List[int]): A list of block number of commitment.
    - repo_ids (List[str]): A list of repo id of the model on hugging face.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given models.
    """
    bt.logging.info(f"♏ Evaluating models ...")
    # Get the last commit time of models
    last_commit_time = get_last_commit_time(responses)
    latest_time = float('inf')

    # Calculate loss of models
    loss_of_models = get_loss(model_paths, uids)

    # Sort the list by the value, keeping track of original indices
    sorted_loss = sorted((value, idx) for idx, value in loss_of_models)
    
    loss_of_models = get_loss(model_paths, uids) # Loss values of models
    ip_counts = Counter(ips) # Count occurrences of each ip
    weight_best_miner = 30 # Weight for the best miner
    ip_limitation = 15 # Allowed maximum occurrences
    alpha = 0.98 # Step size used for calculating reward movement

    # Rank of models
    loss_indices = list(enumerate(loss_of_models)) # Combine loss values with their corresponding indices
    sorted_indices = sorted(loss_indices, key=lambda x: (x[1], commit_blocks[x[0]])) # Loss first, then time
    ranks = {} # A dictionary to store ranks

    # Assign ranks to the sorted indices
    for rank, pair in enumerate(sorted_indices):
        idx = pair[0]
        ranks[idx] = rank
        if rank and repo_ids[idx] == repo_ids[idx - 1]:
            ranks[idx] = -1

    alpha_A = 0.8
    alpha_B = 0.9

    for idx, loss_of_model in enumerate(loss_of_models):
        count_miners_same_ip = ip_counts[ips[idx]] # Count of miners with the same ip address
        rank = ranks[idx] # Rank of the model

        if rank == -1 or loss_of_model == float('inf') or commit_blocks[idx] == float('inf'):
            reward = 0
        elif rank == 0:
            reward = weight_best_miner
        elif rank < group_A_rank:
            reward = alpha_A ** rank
        else:
            reward = (alpha_A ** group_A_rank) * (alpha_B ** (rank - group_A_rank))
        rewards.append(reward)

    return torch.FloatTensor(rewards)
