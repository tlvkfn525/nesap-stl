"""
Testing script for MovingMNIST examples
"""

# System
import math
import os
import argparse
import logging

# Externals
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Locals
from datasets import moving_mnist
from trainers import get_trainer
from models import predrnn_pp
from utils.logging import config_logging
from utils.distributed import init_workers, try_barrier

from torch.utils.data import DataLoader

from PIL import Image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/hello.yaml',
            help='YAML configuration file')
    add_arg('-d', '--distributed-backend', choices=['mpi', 'nccl', 'nccl-lsf', 'gloo'],
            help='Specify the distributed backend to use')
    add_arg('--gpu', type=int,
            help='Choose a specific GPU by ID')
    add_arg('--ranks-per-node', type=int, default=8,
            help='Specifying number of ranks per node')
    add_arg('--rank-gpu', action='store_true',
            help='Choose GPU according to local rank')
    add_arg('--resume', action='store_true',
            help='Resume training from last checkpoint')
    add_arg('-v', '--verbose', action='store_true',
            help='Enable verbose logging')
    add_arg('-n', '--ntest', type=int, default=16,
            help='Number of test dataset')
    add_arg('-t', '--threshold', type=float, default=0.015,
            help='Threshold of correctness')
    return parser.parse_args()

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_test_dataloader(data_dir, n_test=None, **kwargs):
    test_dataset = moving_mnist.MovingMNIST(os.path.join(data_dir, 'moving-mnist-test.npz'),
                                            n_samples=n_test)
    test_loader = DataLoader(test_dataset)
    return test_loader

def test(model, device, test_loader, loss_config, threshold):
    logging.info('test with %i test data', len(test_loader.dataset))
    model.eval()
    test_loss = 0
    correct = 0
    Loss = getattr(torch.nn, loss_config.pop('name'))
    loss_func = Loss(**loss_config)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # batch = batch.to(device)
            model.zero_grad()
            batch_input, batch_target = batch[:,:-1], batch[:,1:]
            batch_output = model(batch_input)
            batch_loss = loss_func(batch_output, batch_target).item()
            logging.info('element %i loss %.3f', i, batch_loss)
            if batch_loss < threshold:
                correct += 1
            test_loss += batch_loss

    test_loss /= len(test_loader.dataset)

    logging.info('Test set: Average loss: %.4f, Accuracy: %i/%i (%.0f%%)\n', 
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))


def main():
    """Main function"""

    # Initialization
    args = parse_args()
    rank, n_ranks = init_workers(args.distributed_backend)

    # Load configuration
    config = load_config(args.config)

    # Prepare output directory
    output_dir = config.get('output_dir', None)
    if output_dir is not None:
        output_dir = os.path.expandvars(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = (os.path.join(output_dir, 'out_%i.log' % rank)
                if output_dir is not None else None)
    config_logging(verbose=args.verbose, log_file=log_file, append=args.resume)
    logging.info('Initialized rank %i out of %i', rank, n_ranks)
    try_barrier()
    if rank == 0:
        logging.info('Configuration: %s' % config)

    # Load the datasets
    test_data_loader = get_test_dataloader(**config['data'], n_test=args.ntest)

    # Load the trainer
    gpu = (rank % args.ranks_per_node) if args.rank_gpu else args.gpu
    if gpu is not None:
        logging.info('Using GPU %i', gpu)

    model = predrnn_pp.PredRNNPP()
    checkpoint = torch.load(**config['test'])
    model.load_state_dict(checkpoint['model'])
    loss_config = config['loss']
    test(model, gpu, test_data_loader, loss_config, threshold=args.threshold)

if __name__ == '__main__':
    main()
