"""
Testing script for MovingMNIST examples
"""

# System
from models.cnn import CNNClassifier
import os
import argparse
import logging

# Externals
import yaml
import torch

# Locals
from datasets import moving_mnist, get_data_loaders
from trainers import get_trainer
from models import predrnn_pp
from utils.logging import config_logging
from utils.distributed import init_workers, try_barrier

from torch.utils.data import DataLoader

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
    add_arg('-n', '--ntest', type=int, default=128,
            help='Number of test dataset')
    add_arg('-c', '--checkpoint', type=int, default=12,
            help='Checkpoint number')
    add_arg('-t', '--threshold', type=float, default=0.01,
            help='Threshold of correctness')
    return parser.parse_args()

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_test_dataloader(data_dir, batch_size, n_test=None, **kwargs):
    if data_dir is not None:
        data_dir = os.path.expandvars(data_dir)
        os.makedirs(data_dir, exist_ok=True)
    test_dataset = moving_mnist.MovingMNIST(os.path.join(data_dir, 'moving-mnist-test.npz'),
                                            n_samples=n_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
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
            batch = batch.to(device)
            model.zero_grad()
            batch_input, batch_target = batch[:,:-1], batch[:,1:]
            batch_output = model(batch_input)
            batch_loss = loss_func(batch_output, batch_target).item()
            logging.info('batch %i loss %.4f', i, batch_loss)
            if batch_loss <= threshold:
                correct += 1
            test_loss += batch_loss

    test_loss /= len(test_loader)

    logging.info('Test set: Average loss: %.4f, Accuracy: %i/%i (%.0f%%)\n', 
        test_loss, correct, len(test_loader),
        100. * correct / len(test_loader))


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

    gpu = (rank % args.ranks_per_node) if args.rank_gpu else args.gpu
    if gpu is not None:
        logging.info('Using GPU %i', gpu)
        device = torch.device("cuda:" + str(gpu))
    else:
        logging.info('Using CPU')
        device = torch.device("cpu")

    # Load the datasets
    test_data_loader = get_test_dataloader(**config['data'], n_test=args.ntest)
    distributed = args.distributed_backend is not None

    do_classify = config.get('classify_data', None)
    model = predrnn_pp.PredRNNPP(classify=(do_classify is not None))
    checkpoint_idx = str(args.checkpoint) if args.checkpoint >= 100 else ('0' + str(args.checkpoint))
    checkpoint = torch.load(os.path.join(output_dir, 'checkpoints', 'checkpoint_' + checkpoint_idx + '.pth.tar'), map_location=("cuda:" + str(gpu)))
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    loss_config = config['loss']

    if do_classify is not None:
        train_data_loader, valid_data_loader = get_data_loaders(
            distributed=distributed, **config['classify_data'])
        # TODO: require arguments to create class instance
        classify_model = CNNClassifier()
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(classify_model.parameters(), lr=0.001)
        for epoch in range(int(config['train']['n_epochs'])):
            train_avg_cost = 0
            valid_avg_cost = 0

            for batch, label in train_data_loader:
                batch = batch.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                model_output = model(batch)
                hypothesis = classify_model(model_output)
                cost = criterion(hypothesis, label)
                cost.backward()
                optimizer.step()

                train_avg_cost += cost / len(train_data_loader)
            logging.info('[epoch {:>4}: train] cost = {:>.9}'.format(epoch + 1, train_avg_cost))

            for batch, label in valid_data_loader:
                batch = batch.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                model_output = model(batch)
                hypothesis = classify_model(model_output)
                cost = criterion(hypothesis, label)

                valid_avg_cost += cost / len(valid_data_loader)
            logging.info('[epoch {:>4}: valid] cost = {:>.9}'.format(epoch + 1, valid_avg_cost))
    else:
        test(model, device, test_data_loader, loss_config, threshold=args.threshold)

if __name__ == '__main__':
    main()
