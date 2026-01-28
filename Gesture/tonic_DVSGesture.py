
"""
Loading datasets from neuromorphic sensors is made super simple thanks to Tonic, which works much like PyTorch vision.
"""
import sys, click, torch, torchvision, tonic, tonic.transforms as transforms, snntorch as snn, torch.nn as nn, os, matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from tonic import MemoryCachedDataset
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

surrogate_fns = {
    "atan" : surrogate.atan(),
    "sigm" : surrogate.sigmoid(),
    "fsigm" : surrogate.fast_sigmoid(),
}


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
    data_path='/tmp/data/DVSGesture'
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, time_window=time_window_us)
        ])
    trainset = tonic.datasets.DVSGesture(save_to=data_path, transform=frame_transform, train=True)
    testset = tonic.datasets.DVSGesture(save_to=data_path, transform=frame_transform, train=False)
    
    """
    Since the original data is stored in a format that is slow to read, Tonic will convert and cache the transformed data for
    faster loading in subsequent runs. That means that once files are loaded from the original dataset, they are written to
    the disk. 
    If you have a large amount of RAM available, you can speed up dataloading further by caching to main memory instead of to disk.

    """
    return MemoryCachedDataset(trainset), MemoryCachedDataset(testset)

def get_net(device, input_depth, nclasses):
    """
    The convolutional network architecture to be used is the one from:
    K. Stewart, G. Orchard, S. B. Shrestha, e E. Neftci, «Online Few-Shot Gesture Learning on a
    Neuromorphic Processor», IEEE J. Emerg. Sel. Topics Circuits Syst., vol. 10, fasc. 4, pp. 512–521, 
    dic. 2020, doi: 10.1109/JETCAS.2020.3032058.
    """
    spike_grad = surrogate.atan()
    decay_rate = 0.5
    net = nn.Sequential(nn.MaxPool2d(4),  # downsample input size from 128x128 to 32x32
                        nn.Conv2d(in_channels = input_depth, out_channels = 16, kernel_size = 5, stride = 1, padding=2),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
                        nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding=1),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
                        nn.Flatten(),
                        nn.Linear(in_features = 2048, out_features = 512),
                        snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
                        nn.Linear(in_features = 512, out_features = nclasses),
                        snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True, output=True)).to(device)
    return net

def forward_pass(snn_model, samples):
    mem_rec = []
    spk_rec = []
    utils.reset(snn_model)  # resets hidden states for all LIF neurons in net
    for step in range(samples.size(0)):
        spk_out, mem_out = snn_model(samples[step])
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
    return torch.stack(spk_rec), torch.stack(mem_rec)

def train_loop(device, trainloader, net, optimizer, loss_fn, num_epochs, num_iters, filename='train_set_accuracy.pdf'):
    loss_hist = [float("inf")]
    acc_hist = [0]
    for _ in trange(num_epochs, desc = f"Train loop. Loss: {loss_hist[-1]:.2f}. Accuracy: {acc_hist[-1] * 100:.2f}%", file=sys.stdout, leave = True, bar_format="{desc:50} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
        for i, (data, targets) in tqdm(enumerate(iter(trainloader)), desc = "Iteration: ", total=num_iters, file=sys.stdout, leave = True, bar_format="{desc:50} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
            data = data.to(device)
            targets = targets.to(device)
            net.train()
            spk_rec, _ = forward_pass(net, data)
            
            loss_val = loss_fn(spk_rec, targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_hist.append(loss_val.item())
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)

            if i == num_iters:
                break  # training loop breaks after N iterations
      
    fig = plt.figure(facecolor="w")
    plt.plot(acc_hist)
    plt.plot(acc_hist,label="Accuracy")
    plt.plot(loss_hist,label="Loss")
    plt.title("Train Set Accuracy and Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.savefig(filename)
    
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
  return acc/total

@click.group()
def cli():
    pass

@cli.command()
@click.argument('outfile', type=click.Path(exists=False, readable=True, dir_okay=False))
@click.option('-t', '--time', type=int, default=20000, help='Time window. --time_window=1000 integrates events into 1000us bins. Larger time window allows for smaller tensors.')
@click.option('-b', '--batchsize', type=int, default=128, help='batch size')
@click.option('-e','--epochs', type=int, default=1, help='number of training epochs')
@click.option('-i','--iterations', type=int, default=50, help='number of iterations')
@click.option('-l','--learningrate', type=float, default=2e-2, help='learning rate')
def train(outfile, time, batchsize, epochs, iterations, learningrate):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    ## Getting the dataset
    trainset, _ = get_dataset(time)
    print(f"Training samples: {len(trainset)}")

    """
    Because event recordings have different lengths, we are going to provide a collation function
    tonic.collation.PadTensors() that will pad out shorter recordings to ensure all samples in a batch have the same dimensions.

    Note: we don't need to specify the simulation time steps for the SNN as a parameter here.
    The number of time steps will be set to that of the longest recording in the mini-batch, and all other samples will be
    padded with zeros to match it.

    The structure of the input data must be [num_steps x batch_size x (input dimensions)] e.g. (310 x 128 x 2 x 32 x 32). 
    Therefore, the batch_first=False is of utter importance
    """
    trainloader = DataLoader(trainset, batch_size=batchsize, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    data_it, _ = next(iter(trainloader))
    print(f"Spike data size: {data_it.size()}")

    net = get_net(device, data_it.size()[2], 11)

    print(f"Performing the training loop with {epochs} epochs.\nBatch size is {batchsize}.\nLearning rate is set to {learningrate}")
    train_loop(device, trainloader, net,
            optimizer = torch.optim.Adam(net.parameters(), lr = learningrate, betas=(0.9, 0.999)), 
            loss_fn =  SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2), 
            num_epochs = epochs, 
            num_iters = iterations)
    torch.save(net.state_dict(), outfile)

@cli.command()
@click.argument('infile', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.option('-t', '--time', type=int, default=20000, help='Time window. --time_window=1000 integrates events into 1000us bins. Larger time window allows for smaller tensors.')
def test(infile, time):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
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
    testloader = DataLoader(testset, collate_fn=tonic.collation.PadTensors(batch_first=False))
    data_it, _ = next(iter(testloader))
    print(f"Spike data size: {data_it.size()}")

    net = get_net(device, data_it.size()[2], 11)

    print(net)

    print(f"Loading weights from {infile}")
    net.load_state_dict(torch.load(infile, weights_only=True))
    net.eval()
    test_accuracy = compute_accuracy(device, testloader, net)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%\n")

    
if __name__ == '__main__':
    cli()