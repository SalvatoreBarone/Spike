#!python
import click, torch, tonic, snntorch, torch.multiprocessing as mp, warnings, json, itertools
warnings.filterwarnings("ignore")

from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import spikegen

from distutils.dir_util import mkpath
from torch.distributed import destroy_process_group
from src.snn import forward_pass, forward_pass_2nd, compute_accuracy, plot_spk_cur_mem_spk, plot_spk_mem_spk
from src.DDPTrainers import SimpleDDPTrainer, ddp_setup
from src.datasets.DVSGesture import DVSGestureTimeBins
from src.architectures.CNN import MP4_16C5_MP2_32C3_MP2_2048FC512_512FCnclasses as CNN

from tabulate import tabulate


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
    
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    acc_fn = SF.accuracy_rate
    optimizer = torch.optim.Adam(
        cnn.model.parameters(),
        lr = kwargs.get("learning_rate", 2e-2),
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.9) 


    trainer = SimpleDDPTrainer(
        cnn.model, forward_pass, dataset.train_loader, dataset.test_loader, loss_fn, acc_fn, optimizer, scheduler, rank, save_every, outdir
    )
    trainer.train(kwargs.get("epochs", 1))
    destroy_process_group()


def ablation_1st_fn(
    # DDP parameters
    rank: int,
    world_size: int,
    
    # Other parameters
    outdir: str,
    
    ## Additional keyword arguments for neuron model
    kwargs
):
    ddp_setup(rank, world_size)
    
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    acc_fn = SF.accuracy_rate
    
    results = []
    
    filter_time = kwargs.get("filter_time", [1000, 2000, 3000, 4000, 5000, 7000,  10000]) 
    n_time_bins = kwargs.get("n_time_bins", [32, 64, 128, 256])

    beta = kwargs.get("beta", [0.9, 0.5, "learned"])
    threashold = kwargs.get("threshold", [1, "learned"])
    reset = kwargs.get("reset", ["subtract", "zero"]) 
    
    for filter, bins, b, t, r in itertools.product(filter_time, n_time_bins, beta, threashold, reset):
        
        dataset = DVSGestureTimeBins(
            batch_size = kwargs.get("batch_size", 32),
            filter_time = filter,
            n_time_bins = bins,
            rank=rank
        ) 
        
        kwargs["beta"] = beta[0] if b == "learned" else b
        kwargs["learn_beta"] = (b == "learned")
        kwargs["threshold"] = threashold[0] if t == "learned" else t
        kwargs["learn_threshold"] = (t == "learned")
        kwargs["reset"] = r
        
        if rank == 0:
            print(f"Filter time: {filter}, bins: {bins}, kwargs: {json.dumps(kwargs, indent=4)}")
            
        workdir = f"{outdir}/filter_{filter}_bins_{bins}_beta_{b}_threshold_{t}_reset_{r}"
        mkpath(workdir)
        
        cnn = CNN(
            dataset.ishape[2],
            dataset.nclasses,
            snntorch.Leaky, 
            kwargs = kwargs)
        
        optimizer = torch.optim.Adam(
            cnn.model.parameters(),
            lr = kwargs.get("learning_rate", 2e-2),
            betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.9) 

        
        trainer = SimpleDDPTrainer(
            cnn.model, forward_pass, dataset.train_loader, dataset.test_loader, loss_fn, acc_fn, optimizer, scheduler, rank, 5, workdir
        )
        
        trainer.train(kwargs.get("epochs", 30))

        results.append([filter, bins, b, t, r, trainer.train_acc_hist[-1], trainer.test_acc_hist[-1], trainer.loss_hist[-1]])
                
    if rank == 0:
        with open(f"{outdir}/ablation_results.md", "w") as f:
             f.write(tabulate(results, headers=["Filter Time", "Time Bins", "Beta", "Threshold", "Reset", "Train Acc", "Test Acc", "Final Loss"],  tablefmt="github", floatfmt=".4f"))         
                
    destroy_process_group() 
        
        
def ablation_2nd_nf(
    # DDP parameters
    rank: int,
    world_size: int,
    
    # Other parameters
    outdir: str,
    
    ## Additional keyword arguments for neuron model
    kwargs
):
    ddp_setup(rank, world_size)
    
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    acc_fn = SF.accuracy_rate
    
    results = []
    
    filter_time = kwargs.get("filter_time", [1000, 2000, 3000, 4000, 5000, 7000,  10000]) 
    n_time_bins = kwargs.get("n_time_bins", [32, 64, 128, 256])

    alpha = kwargs.get("alpha", [0.9, 0.5, "learned"])  
    beta = kwargs.get("beta", [0.9, 0.5, "learned"])
    threashold = kwargs.get("threshold", [1, "learned"])
    reset = kwargs.get("reset", ["subtract", "zero"]) 
    
    for filter, bins, a, b, t, r in itertools.product(filter_time, n_time_bins, alpha, beta, threashold, reset):
        
        dataset = DVSGestureTimeBins(
            batch_size = kwargs.get("batch_size", 32),
            filter_time = filter,
            n_time_bins = bins,
            rank=rank
        ) 
        
        kwargs["alpha"] = alpha[0] if a == "learned" else a
        kwargs["learn_alpha"] = (a == "learned") 
        kwargs["beta"] = beta[0] if b == "learned" else b
        kwargs["learn_beta"] = (b == "learned")
        kwargs["threshold"] = threashold[0] if t == "learned" else t
        kwargs["learn_threshold"] = (t == "learned")
        kwargs["reset"] = r
        
        if rank == 0:
            print(f"Filter time: {filter}, bins: {bins}, kwargs: {json.dumps(kwargs, indent=4)}")
        
        workdir = f"{outdir}/filter_{filter}_bins_{bins}_alpha_{a}_beta_{b}_threshold_{t}_reset_{r}"
        mkpath(workdir)
        
        cnn = CNN(
            dataset.ishape[2],
            dataset.nclasses,
            snntorch.Synaptic, 
            kwargs = kwargs)
        
        optimizer = torch.optim.Adam(
            cnn.model.parameters(),
            lr = kwargs.get("learning_rate", 2e-2),
            betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.9) 

        trainer = SimpleDDPTrainer(
            cnn.model, forward_pass_2nd, dataset.train_loader, dataset.test_loader, loss_fn, acc_fn, optimizer, scheduler, rank, 5, workdir
        )
        
        trainer.train(kwargs.get("epochs", 30))

        results.append([filter, bins, a, b, t, r, trainer.train_acc_hist[-1], trainer.test_acc_hist[-1], trainer.loss_hist[-1]])
                
    if rank == 0:
        with open(f"{outdir}/ablation_results.md", "w") as f:
             f.write(tabulate(results, headers=["Filter Time", "Time Bins", "Alpha", "Beta", "Threshold", "Reset", "Train Acc", "Test Acc", "Final Loss"],  tablefmt="github", floatfmt=".4f"))         
                
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
        "reset" : "zero" if resetzero else "subtract",
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


@cli.command()
@click.argument("outdir", type=click.Path(readable=True, writable=True, dir_okay=True, file_okay=False))
@click.argument("order", type=click.Choice(["1", "2"]))
@click.option("-b", "--batchsize", type=int, default=128, help="batch size")
@click.option("-e", "--epochs", type=int, default=1, help="Number of training epochs")
@click.option("-l", "--learningrate", type=float, default=2e-2, help="learning rate")

@click.option("-f", "--filter_time", type=int, default=None, help="Denoising filter time in microseconds. Smaller filter_time will filter more events.")
@click.option("-t", "--bins", type=int, default=None, help="Number of time bins to integrate events into. Larger number of bins allows for more temporal precision but also larger tensors.")

@click.option("-a", "--alpha", type=float, default=None, help="Alpha parameter for the 2nd-order neuron model.")
@click.option("-b", "--beta", type=float, default=None, help="Beta parameter for the 2nd-order neuron model.")
@click.option("-t", "--threshold", type=float, default=None, help="Threshold parameter for the 2nd-order neuron model.")
@click.option("-r", "--reset", type=click.Choice(["subtract", "zero"]), default=None, help="Reset mechanism for the 2nd-order neuron model.")

def ablation(outdir, order, batchsize, epochs, learningrate, filter_time, bins, alpha, beta, threshold, reset):

    mkpath(outdir)
     
    gpu_count = get_gpu_count()
   
    kwargs = { 
        "epochs"          : epochs,
        "batch_size"      : batchsize, 
        "learning_rate"   : learningrate
        }
    
    if filter_time is not None:
        kwargs["filter_time"] = [filter_time]
    if bins is not None:
        kwargs["n_time_bins"] = [bins]
    
    if alpha is not None:
        kwargs["alpha"] = [alpha]
    if beta is not None:
        kwargs["beta"] = [beta]
    if threshold is not None:
        kwargs["threshold"] = [threshold]
    if reset is not None:
        kwargs["reset"] = [reset]

    mp.spawn(
        ablation_1st_fn if order == "1" else ablation_2nd_nf,
        args=(gpu_count, outdir, kwargs),
        nprocs = gpu_count,
        join = True
    )
    
    
@cli.command()
@click.argument("order", type=click.Choice(["1", "2"]))
@click.option("-a", "--alpha", type=float, default=0.9, help="Alpha parameter for the 2nd-order neuron model.")
@click.option("-b", "--beta", type=float, default=0.9, help="Beta parameter for the 2nd-order neuron model.")
@click.option("-t", "--threshold", type=float, default=1.0, help="Threshold parameter for the 2nd-order neuron model.")
@click.option("-r", "--reset", type=click.Choice(["subtract", "zero"]), default="subtract", help="Reset mechanism for the 2nd-order neuron model.")
@click.option("-n", "--num_steps", type=int, default=200, help="Number of time steps to simulate.")
@click.option("-v", "--voltage", type=float, default=0.2, help="Voltage for the spiking input.")
@click.option("-p", "--period", type=int, default=9, help="Period of the spiking input in time steps.")
def simulate(order, alpha, beta, threshold, reset, num_steps, voltage, period):
    # Periodic spiking input, spk_in = 0.2 V
    spk_period = torch.cat((torch.ones(1)*voltage, torch.zeros(period)), 0)
    
    length = num_steps // (period + 1) + 1
    spk_in = spk_period.repeat(length)

    if order == "1":
        # Initialize LIF neuron
        lif1 = snntorch.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset)

        # Initialize hidden state and output
        mem = lif1.reset_mem()
        spk_out = torch.zeros(1)
        
        mem_rec = []
        spk_rec = []
        # Simulate neurons
        for step in range(num_steps):
            spk_out, mem = lif1(spk_in[step], mem)
            spk_rec.append(spk_out)
            mem_rec.append(mem)

        # convert lists to tensors
        spk_rec = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)

        plot_spk_mem_spk(spk_in, mem_rec, spk_rec, threshold, outfile=f"{order}_{beta}_{threshold}_{reset}_P{period+1}_simulation.pdf")
        
    else:
        # Initialize 2nd-order LIF neuron
        lif1 = snntorch.Synaptic(alpha=alpha, beta=beta, threshold=threshold, reset_mechanism=reset)

        # Initialize hidden states and output
        syn, mem = lif1.reset_mem()
        spk_out = torch.zeros(1)
        
        syn_rec = []
        mem_rec = []
        spk_rec = []
        # Simulate neurons
        for step in range(num_steps):
            spk_out, syn, mem = lif1(spk_in[step], syn, mem)
            spk_rec.append(spk_out)
            syn_rec.append(syn)
            mem_rec.append(mem)

        # convert lists to tensors
        spk_rec = torch.stack(spk_rec)
        syn_rec = torch.stack(syn_rec)
        mem_rec = torch.stack(mem_rec)

        plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec, threshold, outfile=f"{order}_{alpha}_{beta}_{threshold}_{reset}_P{period+1}_simulation.pdf")

if __name__ == "__main__":
    cli()
