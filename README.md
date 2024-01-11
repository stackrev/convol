<div align="center">

# Healthcare-Subnet
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---
- [Introduction](#introduction)
- [Installation](#installation)
  - [Before you proceed](#before-you-proceed)
  - [Install](#install)
- [Running](#running)
  - [Running subtensor locally](#running-subtensor-locally)
  - [Running miner](#running-miner)
  - [Running validator](#running-validator)
- [License](#license)

---
## Introduction

A groundbreaking healthcare subnet on the Bittensor network!
In this innovative system, validators seamlessly transmit medical images to miners, who, in turn, predict diseases with precision. Miners earn rewards based on the accuracy of their predictions and continually enhance their models through training. Our healthcare subnet stands at the forefront of collaborative, AI-driven healthcare solutions, promising advancements in disease diagnosis and contributing to the evolution of medical research.

## Installation

### Before you proceed
Before you proceed with the installation of the subnet, note the following: 

- Use these instructions to run your subnet locally for your development and testing, or on Bittensor testnet or on Bittensor mainnet. 
- **IMPORTANT**: We **strongly recommend** that you first run your subnet locally and complete your development and testing before running the subnet on Bittensor testnet. Furthermore, make sure that you next run your subnet on Bittensor testnet before running it on the Bittensor mainnet.
- You can run your subnet either as a subnet validator or as a subnet miner. 
- **IMPORTANT:** Make sure you are aware of the minimum compute requirements for your subnet. See the [Minimum compute YAML configuration](./min_compute.yml).

### Install

#### Bittensor

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/opentensor/bittensor/master/scripts/install.sh)"
```

#### Clone the repository from Github
```bash
git clone https://github.com/bthealthcare/healthcare-subnet.git
```

#### Install package dependencies for the repository
```bash
cd healthcare-subnet
apt install python3-pip -y
python3 -m pip install -e .
```

#### Install `pm2`
```bash
apt update && apt upgrade -y
apt install nodejs npm -y
npm i -g pm2
```
---

## Running

### Running subtensor locally

#### Install Docker
```bash
apt install docker.io -y
apt install docker-compose -y
```

#### Run Subtensor locally
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker-compose up --detach
```

### Running miner
In this innovative healthcare subnet, miners play a crucial role in contributing to disease diagnosis by predicting from medical images. Through continuous training, miners strive to improve their models, with more accurate models earning substantial rewards. Miners have the flexibility to adapt and enhance the structure of their models, datasets, and other factors influencing model accuracy. This collaborative effort aims to advance disease prediction and underscores the vital role miners play in shaping the future of medical diagnostics.

#### Download the dataset for model training
```bash
python3 healthcare/dataset/downloader.py
```

#### Run the miner with `pm2`
```bash
# To run the miner
pm2 start neurons/miner.py --name miner --interpreter python3 -- 
    --netuid # the subnet netuid, default = 
    --subtensor.network # the bittensor chain endpoint, default = finney, local, test (highly recommend running subtensor locally)
    --wallet.name # your miner wallet, default = default
    --wallet.hotkey # your validator hotkey, default = default
    --logging.debug # run in debug mode, alternatively --logging.trace for trace mode
    --num_epochs # the number of training epochs (-1 is infinite), default = -1
    --batch_size # the number of data points processed in a single iteration, default = 32
    --threshold # the predefined cutoff value that is used to determine which labels should be selected based on their corresponding scores, default = 0.2
    --save_model_period # the period of batches during which the model is saved, default = 30
    --restart # if set, miners will start the training from scratch, default = False
```

- Example 1 (with default values):
```bash
pm2 start neurons/miner.py --name miner --interpreter python3 -- --wallet.name default --wallet.hotkey default --logging.debug
```

- Example 2 (with custom values):
```bash
pm2 start neurons/miner.py --name miner --interpreter python3 --
    --subtensor.network local
    --wallet.name default
    --wallet.hotkey default
    --logging.debug
    --num_epochs 10
    --batch_size 256
    --threshold 0.5
    --restart True
```

### Running validator
Validators play a pivotal role in evaluating miner's models by periodically sending diverse images for assessment. They meticulously score miners based on their responses, contributing to the ongoing refinement of models and ensuring the highest standards of performance and accuracy in our collaborative network.

#### Asking demon(discord) to provide the dataset
Validators are kindly encouraged to request the dataset from Demon (Discord username) before initiating the process. Please feel free to establish communication with the subnet owner and inquire if they would be willing to provide the dataset.

#### Run the validator
```bash
# To run the validator
pm2 start neurons/validator.py --name validator --interpreter python3 -- 
    --netuid # the subnet netuid, default = 
    --subtensor.network # the bittensor chain endpoint, default = finney, local, test (highly recommend running subtensor locally)
    --wallet.name # your miner wallet, default = default
    --wallet.hotkey # your validator hotkey, default = default
    --logging.debug # run in debug mode, alternatively --logging.trace for trace mode
    --vpermit_tao_limit # the maximum number of TAO allowed to query a validator with a vpermit, default = 4096
```

- Example 1 (with default values):
```bash
pm2 start neurons/validator.py --name validator --interpreter python3 -- --wallet.name default --wallet.hotkey default --logging.debug
```

- Example 2 (with custom values):
```bash
pm2 start neurons/validator.py --name validator --interpreter python3 --
    --subtensor.network local
    --wallet.name default
    --wallet.hotkey default
    --logging.debug
    --vpermit_tao_limit 1024
```

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
```