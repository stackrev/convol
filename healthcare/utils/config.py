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
import torch
import argparse
import bittensor as bt
from loguru import logger


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        logger.level("EVENTS", no=38, icon="📝")
        logger.add(
            os.path.join(config.neuron.full_path, "events.log"),
            rotation=config.neuron.events_retention_size,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            level="EVENTS",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )


def add_args(cls, parser):
    if "BaseNeuron" in cls.__name__:
        return
    """
    Adds relevant arguments to the parser for operation.
    """
    # Netuid Arg: The netuid of the subnet to connect to.
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=31)

    neuron_type = (
        "validator" if "miner" not in cls.__name__.lower() else "miner"
    )

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default=neuron_type,
    )

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default="cpu",
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=100,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default="2 GB",
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--neuron.auto_update",
        action="store_true",
        help="If set, we will update the current mechine to the latest one.",
        default=False,
    )

    if neuron_type == "validator":
        parser.add_argument(
            "--neuron.num_concurrent_forwards",
            type=int,
            help="The number of concurrent forwards running at any time.",
            default=1,
        )
        
        parser.add_argument(
            "--neuron.query_time",
            type=int,
            help="The number of steps for a single query.",
            default=2,
        )
        
        parser.add_argument(
            "--neuron.score_update_time",
            type=int,
            help="The number of steps for a single score update.",
            default=10,
        )

        parser.add_argument(
            "--neuron.sample_size",
            type=int,
            help="The number of miners to query in a single step.",
            default=10,
        )

        parser.add_argument(
            "--neuron.disable_set_weights",
            action="store_true",
            help="Disables setting weights.",
            default=False,
        )

        parser.add_argument(
            "--neuron.moving_average_alpha",
            type=float,
            help="Moving average alpha parameter, how much to add of the new observation.",
            default=0.05,
        )

        parser.add_argument(
            "--neuron.axon_off",
            "--axon_off",
            action="store_true",
            # Note: the validator needs to serve an Axon with their IP or they may
            #   be blacklisted by the firewall of serving peers on the network.
            help="Set this flag to not attempt to serve an Axon.",
            default=False,
        )

        parser.add_argument(
            "--neuron.vpermit_tao_limit",
            type=int,
            help="The maximum number of TAO allowed to query a validator with a vpermit.",
            default=4096,
        )

    else:
        parser.add_argument(
            "--blacklist.force_validator_permit",
            action="store_true",
            help="If set, we will force incoming requests to have a permit.",
            default=False,
        )

        parser.add_argument(
            "--blacklist.allow_non_registered",
            action="store_true",
            help="If set, miners will accept queries from non registered entities. (Dangerous!)",
            default=False,
        )

        parser.add_argument(
            "--num_epochs", 
            type = int, 
            default = -1, 
            help="Number of training epochs (-1 is infinite)"
        )

        parser.add_argument(
            "--batch_size", 
            type = int, 
            default = 32, 
            help="The batch size"
        )

        parser.add_argument(
            "--restart",
            action="store_true",
            default = False, 
            help="If set miners will start the training from scratch."
        )

        parser.add_argument(
            "--threshold",
            default = 0.2,
            help="The predefined cutoff value that is used to determine which labels should be selected based on their corresponding scores."
        )

        parser.add_argument(
            "--save_model_period",
            default = 30,
            help="The period of batches during which the model is saved."
        )

def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)