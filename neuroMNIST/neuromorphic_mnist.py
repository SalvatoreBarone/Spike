
"""
Loading datasets from neuromorphic sensors is made super simple thanks to Tonic, which works much like PyTorch vision.
"""
import torch, torchvision, tonic, tonic.transforms as transforms, snntorch as snn, torch.nn as nn, os, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
from tonic import MemoryCachedDataset
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import functional as SF
from snntorch import utils


def get_MNIST():

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
    data_path='/tmp/data/neuro-mnist'
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=1000)
        ])

    print(f"Sensor size: {tonic.datasets.NMNIST.sensor_size}")

    trainset = tonic.datasets.NMNIST(save_to=data_path, transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to=data_path, transform=frame_transform, train=False)
    
    """
    Since the original data is stored in a format that is slow to read, Tonic will convert and cache the transformed data for
    faster loading in subsequent runs. That means that once files are loaded from the original dataset, they are written to
    the disk. 
    If you have a large amount of RAM available, you can speed up dataloading further by caching to main memory instead of to disk.

    """
    return MemoryCachedDataset(trainset), MemoryCachedDataset(testset)

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
    loss_hist = []
    acc_hist = []
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device)
            targets = targets.to(device)
            net.train()
            spk_rec, _ = forward_pass(net, data)
            
            loss_val = loss_fn(spk_rec, targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_hist.append(loss_val.item())
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")

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


if __name__ == "__main__":
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    ## Getting the dataset
    trainset, testset = get_MNIST()
    print(f"Training samples: {len(trainset)}")
    print(f"Training data size: {trainset[0][0].shape}")
    """
    Because event recordings have different lengths, we are going to provide a collation function
    tonic.collation.PadTensors() that will pad out shorter recordings to ensure all samples in a batch have the same dimensions.

    Note: we don't need to specify the simulation time steps for the SNN as a parameter here.
    The number of time steps will be set to that of the longest recording in the mini-batch, and all other samples will be
    padded with zeros to match it.

    The structure of the input data must be [num_steps x batch_size x (input dimensions)] e.g. (310 x 128 x 2 x 32 x 32). 
    Therefore, the batch_first=False is of utter importance
    """
    batch_size = 128
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

    data_it, targets_it = next(iter(trainloader))
    print(f"Spike data size: {data_it.size()}")
        
    """
    The convolutional network architecture to be used is: 12C5-MP2-32C5-MP2-800FC10
    12C5 is a 5x5 convolutional kernel with 12 filters
    MP2 is a 2x2 max-pooling function
    22C5 is a 5x5 convolutional kernel with 32 filters
    800FC10 is a fully-connected layer that maps 800 neurons to 10 outputs
    """
    spike_grad = surrogate.atan()
    decay_rate = 0.5
    net = nn.Sequential(nn.Conv2d(2, 12, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
                        nn.Conv2d(12, 32, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
                        nn.Flatten(),
                        nn.Linear(32*5*5, 10),
                        snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)
    
    print(f"Network model:\n{net}")

    """
    The approach below uses the Mean Square Error Spike Count Loss, which aims to elicit spikes from the correct class 80% of the
    time, and 20% of the time from incorrect classes. Encouraging incorrect neurons to fire could be motivated to avoid dead
    neurons.

    Training neuromorphic data is expensive as it requires sequentially iterating through many time steps (approximately 300 time steps in the N-MNIST dataset).

    Warning: the following simulation will take a while. Go make yourself a coffee, or ten.
    """
    model_filename = "neuro-MNIST.pth"
    if not os.path.exists(model_filename):
        num_epochs = 1
        num_iters = 50
        learning_rate = 2e-2
        print(f"Performing the training loop with {num_epochs} epochs.\nBatch size is {batch_size}.\nLearning rate is set to {learning_rate}")
        train_loop(device, trainloader, net,
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999)), 
                loss_fn =  SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2), 
                num_epochs = num_epochs, 
                num_iters = num_iters)
        torch.save(net.state_dict(), model_filename)
    else:
        print(f"Loading weights from {model_filename}")
        net.load_state_dict(torch.load(model_filename, weights_only=True))

    net.eval()
    test_accuracy = compute_accuracy(device, testloader, net)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%\n")