"""
Loading datasets from neuromorphic sensors is made super simple thanks to Tonic, which works much like PyTorch vision.
"""

import click, torch, tonic, tonic.transforms as transforms, snntorch, torch.nn as nn, os
from distutils.dir_util import mkpath
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tonic import MemoryCachedDataset
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

from src.DDPTrainers import SimpleDDPTrainer, ddp_setup

def get_dataset(time_window_us):
    """
    Let's start by loading the neuromorphic version of the MNIST dataset, called N-MNIST.

    SNNs don’t take lists of events as input. The raw data must be converted into a suitable representation, such as a tensor.
    We apply two transforms to the data:
        1. Denoising to removes isolated, one-off events. If no event occurs within a neighbourhood of 1 pixel across
        filter_time microseconds, the event is filtered. Smaller filter_time will filter more events.
        2. ToFrame to convert the event list into a tensor of shape (time_bins, height, width). This is because neuromorphic
        camera sensor which has been used to capture the images has a temporal resolution of 1 microseconds, which when
        converted into a dense representation, ends up as a very large tensor. time_window=1000 integrates events into 1000us
        bins.

    We split the dataset into training and test sets, applying the frame transform to bot
    """
    data_path = "data/"
    train_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(
                sensor_size=tonic.datasets.DVSGesture.sensor_size,
                time_window=time_window_us,
            ),
            v2.RandomRotation(10),
            
        ]
    )
    trainset = tonic.datasets.DVSGesture(
        save_to=data_path, transform=train_transform, train=True
    )
    test_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(
                sensor_size=tonic.datasets.DVSGesture.sensor_size,
                time_window=time_window_us,
            ),
        ]
    )
    testset = tonic.datasets.DVSGesture(
        save_to=data_path, transform=test_transform, train=False
    )

    """
    Since the original data is stored in a format that is slow to read, Tonic will convert and cache the transformed data for
    faster loading in subsequent runs. That means that once files are loaded from the original dataset, they are written to
    the disk. 
    If you have a large amount of RAM available, you can speed up dataloading further by caching to main memory instead of to disk.

    """
    return MemoryCachedDataset(trainset), MemoryCachedDataset(testset)


def get_model(input_depth, nclasses):
    """
    The convolutional network architecture to be used is the one from:
    K. Stewart, G. Orchard, S. B. Shrestha, e E. Neftci, «Online Few-Shot Gesture Learning on a
    Neuromorphic Processor», IEEE J. Emerg. Sel. Topics Circuits Syst., vol. 10, fasc. 4, pp. 512–521,
    dic. 2020, doi: 10.1109/JETCAS.2020.3032058.
    """
    spike_grad = surrogate.atan()
    decay_rate = 0.5
    return nn.Sequential(
        nn.MaxPool2d(4),  # downsample input size from 128x128 to 32x32
        nn.Conv2d(
            in_channels=input_depth, out_channels=16, kernel_size=5, stride=1, padding=2
        ),
        nn.MaxPool2d(2),
        snntorch.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        snntorch.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
        nn.Flatten(),
        nn.Linear(in_features=2048, out_features=512),
        snntorch.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(in_features=512, out_features=nclasses),
        snntorch.Leaky(
            beta=decay_rate, spike_grad=spike_grad, init_hidden=True, output=True
        ),
    )


def forward_pass(snn_model, samples):
    mem_rec = []
    spk_rec = []
    utils.reset(snn_model)  # resets hidden states for all LIF neurons in net
    for step in range(samples.size(0)):
        spk_out, mem_out = snn_model(samples[step])
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
    return torch.stack(spk_rec), torch.stack(mem_rec)


@click.group()
def cli():
    pass


def training_fn(
    rank: int,
    world_size: int,
    time_window: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    save_every: int,
    outdir: str,
):
    ddp_setup(rank, world_size)

    trainset, testset = get_dataset(time_window)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=False,
        sampler=DistributedSampler(trainset),
    )
    data_it, _ = next(iter(train_loader))
    test_loader = DataLoader(
        testset, collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    data_size = data_it.size()[2]
    nclasses = len(tonic.datasets.DVSGesture.classes)
    model = get_model(data_size, nclasses)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.9) 
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    acc_fn = SF.accuracy_rate

    trainer = SimpleDDPTrainer(
        model, forward_pass, train_loader, test_loader, loss_fn, acc_fn, optimizer, scheduler, rank, save_every, outdir
    )
    trainer.train(epochs)
    destroy_process_group()


@cli.command()
@click.argument("ourdir", type=click.Path(readable=True, writable=True, dir_okay=True, file_okay=False))
@click.option(
    "-t",
    "--time",
    type=int,
    default=20000,
    help="Time window. --time_window=1000 integrates events into 1000us bins. Larger time window allows for smaller tensors.",
)
@click.option("-b", "--batchsize", type=int, default=128, help="batch size")
@click.option(
    "-c", "--checkpoint", type=int, default=5, help="Checkpoint every N epochs"
)
@click.option("-e", "--epochs", type=int, default=1, help="Number of training epochs")
@click.option("-l", "--learningrate", type=float, default=2e-2, help="learning rate")
def train(outdir, time, batchsize, checkpoint, epochs, learningrate):
    """
    Train a model using Pytorch Distributed Data Parallel.
    """
    
    mkpath(outdir)
    
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )
    print(f"Device: {device}")
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"Available CUDA devices: {world_size}")
        for i in range(torch.cuda.device_count()):
            print(
                f"{torch.cuda.get_device_properties(i).name}.\n"
                f"\tTotal memory: {torch.cuda.get_device_properties(i).total_memory//2**20} MB.\n"
                f"\tCapabilities: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
            )

    print(
        f"\nPerforming the training loop with {epochs} epochs."
        f"\nBatch size is {batchsize}."
        f"\nLearning rate is set to {learningrate}\n "
    )

    mp.spawn(
        training_fn,
        args=(world_size, time, batchsize, epochs, learningrate, checkpoint, outdir),
        nprocs=world_size,
    )


@cli.command()
@click.argument("infile", type=click.Path(exists=True, readable=True, dir_okay=False))
@click.option(
    "-t",
    "--time",
    type=int,
    default=20000,
    help="Time window. --time_window=1000 integrates events into 1000us bins. Larger time window allows for smaller tensors.",
)
def test(infile, time):

    def compute_accuracy(device, dataloader, snn_model):
        with torch.no_grad():
            total = 0
            acc = 0
            snn_model.eval()
            dataloader = iter(dataloader)
            for data, targets in dataloader:
                data = data.to(device)
                targets = targets.to(device)
                spk_rec, _ = forward_pass(snn_model, data)
                acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                total += spk_rec.size(1)
        return acc / total

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )
    print(f"Device: {device}")

    ## Getting the dataset
    _, testset = get_dataset(time)
    print(f"Testing samples: {len(testset)}")

    """
    Because event recordings have different lengths, we are going to provide a collation function
    tonic.collation.PadTensors() that will pad out shorter recordings to ensure all samples in a batch have the same dimensions.

    Note: we don't need to specify the simulation time steps for the SNN as a parameter here.
    The number of time steps will be set to that of the longest recording in the mini-batch, and all other samples will be
    padded with zeros to match it.

    The structure of the input data must be [num_steps x batch_size x (input dimensions)] e.g. (310 x 128 x 2 x 32 x 32). 
    Therefore, the batch_first=False is of utter importance
    """
    testloader = DataLoader(
        testset, collate_fn=tonic.collation.PadTensors(batch_first=False)
    )
    data_it, _ = next(iter(testloader))
    print(f"Spike data size: {data_it.size()}")

    net = get_model(device, data_it.size()[2], 11).to(device)

    print(net)

    print(f"Loading weights from {infile}")
    net.load_state_dict(torch.load(infile, weights_only=True))
    net.eval()
    test_accuracy = compute_accuracy(device, testloader, net)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%\n")


if __name__ == "__main__":
    cli()
