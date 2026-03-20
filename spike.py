#!python
import click, torch, tonic, snntorch, torch.multiprocessing as mp, warnings, json
warnings.filterwarnings("ignore")

from snntorch import functional as SF
from distutils.dir_util import mkpath
from torch.distributed import destroy_process_group
from src.snn import forward_pass, compute_accuracy
from src.DDPTrainers import SimpleDDPTrainer, ddp_setup
from src.datasets.DVSGesture import DVSGestureTimeWindow, DVSGestureTimeBins
from src.architectures.CNN import MP4_16C5_MP2_32C3_MP2_2048FC512_512FCnclasses as CNN


def get_gpu_count():
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
                f"{torch.cuda.get_device_properties(i).name}."
                f" Total memory: {torch.cuda.get_device_properties(i).total_memory//2**20} MB."
                f" Capabilities: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
            )
            
    return world_size
    
def training_fn(
    # DDP parameters
    rank: int,
    world_size: int,
    
    # Other parameters
    outdir: str,
    save_every: int,
    
    ## Additional keyword arguments for neuron model
    kwargs
):
    ddp_setup(rank, world_size)

    # dataset = DVSGestureTimeWindow(
    #     batch_size = kwargs.get("batch_size", 64),
    #     filter_time = kwargs.get("filter_time", 10000),
    #     time_window_us = kwargs.get("time_window", 20000)
    # )
    
    dataset = DVSGestureTimeBins(
        batch_size = kwargs.get("batch_size", 32),
        filter_time = kwargs.get("filter_time", 5000),
        n_time_bins = kwargs.get("n_time_bins", 32),
        rank=rank
    )
    
    if (rank == 0):
        print(f"Dataset {dataset.__class__.__name__} loaded. Input shape: {dataset.ishape}, number of classes: {dataset.nclasses}")
        print(f"kwargs: {json.dumps(kwargs, indent=4)}")
    
    cnn = CNN(
        dataset.ishape[2],
        dataset.nclasses,
        kwargs.get("neuron_model", snntorch.Leaky), 
        kwargs = kwargs)
    
    if (rank == 0):
        print(f"Model architecture:\n{cnn.model}")
    
    optimizer = torch.optim.Adam(
        cnn.model.parameters(),
        lr = kwargs.get("learning_rate", 2e-2),
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.9) 
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    acc_fn = SF.accuracy_rate

    trainer = SimpleDDPTrainer(
        cnn.model, forward_pass, dataset.train_loader, dataset.test_loader, loss_fn, acc_fn, optimizer, scheduler, rank, save_every, outdir
    )
    trainer.train(kwargs.get("epochs", 1))
    destroy_process_group()


@click.group()
def cli():
    pass

@cli.command()
@click.argument("outdir", type=click.Path(readable=True, writable=True, dir_okay=True, file_okay=False))
@click.option(
    "-f", "--filter_time", type=int, default=10000, help="Denoising filter time in microseconds. Smaller filter_time will filter more events."
    )
@click.option(
    "-t",
    "--time",
    type=int,
    default=20000,
    help="Time window. --time_window=1000 integrates events into 1000us bins. Larger time window allows for smaller tensors.",
)
@click.option("-d", "--learnbeta", is_flag=True, default=False, help="Use learned decay parameters for the neuron model.")
@click.option("-z", "--resetzero", is_flag=True, default=False, help="Use reset-to-zero mechanism for the neuron model instead of reset-by-subtraction.")
@click.option("-v", "--learnthreshold", is_flag=True, default=False, help="Use learned threshold parameters for the neuron model.")
@click.option("-b", "--batchsize", type=int, default=128, help="batch size")
@click.option(
    "-c", "--checkpoint", type=int, default=5, help="Checkpoint every N epochs"
)
@click.option("-e", "--epochs", type=int, default=1, help="Number of training epochs")
@click.option("-l", "--learningrate", type=float, default=2e-2, help="learning rate")
def train(outdir, filter_time, time, learnbeta, resetzero, learnthreshold, batchsize, checkpoint, epochs, learningrate):
    """
    Train a model using Pytorch Distributed Data Parallel.
    """
    
    
    mkpath(outdir)
    
    gpu_count = get_gpu_count()
   
    kwargs = { 
        "filter_time"     : filter_time,
        "time_window"     : time, 
        "n_time_bins"     : time,
        "learn_beta"      : learnbeta,
        "learn_threshold" : learnthreshold,
        "reset_mechanism" : "zero" if resetzero else "subtract",
        "epochs"          : epochs,
        "batch_size"      : batchsize, 
        "learning_rate"   : learningrate
        }
    
    mp.spawn(
        training_fn,
        args=(gpu_count, outdir, checkpoint, kwargs),
        nprocs = gpu_count,
        join = True
    )
 


@cli.command()
@click.argument("infile", type=click.Path(exists=True, readable=True, dir_okay=False))
@click.option(
    "-f", "--filter_time", type=int, default=10000, help="Denoising filter time in microseconds. Smaller filter_time will filter more events."
    )
@click.option(
    "-t",
    "--time",
    type=int,
    default=20000,
    help="Time window. --time_window=1000 integrates events into 1000us bins. Larger time window allows for smaller tensors.",
)
def test(infile, filter_time, time):

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
    
    kwargs = { 
        "filter_time"     : filter_time,
        "time_window"     : time, 
        "n_time_bins"     : time,
        }

    ## Getting the dataset
    dataset = DVSGestureTimeBins(
        batch_size = kwargs.get("batch_size", 32),
        filter_time = kwargs.get("filter_time", 5000),
        n_time_bins = kwargs.get("n_time_bins", 32),
        sampler = None
    )
    
    print(f"Dataset {dataset.__class__.__name__} loaded. Input shape: {dataset.ishape}, number of classes: {dataset.nclasses}")
    print(f"kwargs: {json.dumps(kwargs, indent=4)}")
    
    cnn = CNN(
        dataset.ishape[2],
        dataset.nclasses,
        kwargs.get("neuron_model", snntorch.Leaky), 
        kwargs = kwargs)
    
    cnn.model.to(device)

    print(cnn.model)

    print(f"Loading weights from {infile}")
    cnn.model.load_state_dict(torch.load(infile, weights_only=True))
    print(f"Weights successfully loaded from {infile}")
    
    cnn.model.eval()
    test_accuracy = compute_accuracy(device, dataset.test_loader, cnn.model)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%\n")


if __name__ == "__main__":
    cli()
