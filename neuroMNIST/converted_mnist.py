
"""
Loading datasets from neuromorphic sensors is made super simple thanks to Tonic, which works much like PyTorch vision.
"""
import torch, torchvision, tonic, tonic.transforms as transforms, snntorch as snn, torch.nn as nn, os, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tonic import DiskCachedDataset, MemoryCachedDataset
from snntorch import surrogate, functional as SF, utils, spikegen



def get_MNIST():
    data_path='/tmp/data/mnist'
    transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)  
    return mnist_train, mnist_test

def forward_pass(snn_model, simulation_steps, samples):
  mem_rec = []
  spk_rec = []
  utils.reset(snn_model)  # resets hidden states for all LIF neurons in net
  spike_data = spikegen.rate(samples, num_steps=simulation_steps)
  #print(f"Spike data size: {spike_data.size()}")
  #exit()
  # the spike data shape is [time_steps, batch_size, input_size], e.g., ([100, 128, 1, 28, 28])
  for step in range(simulation_steps):
      spk_out, mem_out = snn_model(spike_data[step])
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)
  return torch.stack(spk_rec), torch.stack(mem_rec)

def train_loop(device, trainloader, net, simulation_steps, optimizer, loss_fn, num_epochs, num_iters, filename='train_set_accuracy.pdf'):
    loss_hist = []
    acc_hist = []
    for epoch in range(num_epochs):
        for i, (samples, targets) in enumerate(iter(trainloader)):
            samples = samples.to(device)
            targets = targets.to(device)
            net.train()
            spk_rec, _ = forward_pass(net, simulation_steps, samples)
            
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
    
def compute_accuracy(device, dataloader, snn_model, simulation_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    snn_model.eval()
    dataloader = iter(dataloader)
    for data, targets in dataloader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(snn_model, simulation_steps, data)
      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)
  return acc/total

if __name__ == "__main__":
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    simulation_steps = 100
    ## Setting up the static MNIST
    mnist_train, mnist_test = get_MNIST()

    # Create DataLoaders
    batch_size = 128
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

    data_it, targets_it = next(iter(train_loader))
    print(f"Spike data shape: {data_it.size()}")

    """
    The convolutional network architecture to be used is: 12C5-MP2-32C5-MP2-800FC10
    12C5 is a 5x5 convolutional kernel with 12 filters
    MP2 is a 2x2 max-pooling function
    22C5 is a 5x5 convolutional kernel with 32 filters
    800FC10 is a fully-connected layer that maps 800 neurons to 10 outputs
    """
    spike_grad = surrogate.atan()
    decay_rate = 0.5
    net = nn.Sequential(
                        nn.Conv2d(1, 12, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
                        nn.Conv2d(12, 64, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
                        nn.Flatten(),
                        nn.Linear(64*4*4, 10),
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
    
    model_filename = "converted-MNIST.pth"
    if not os.path.exists(model_filename):
        num_epochs = 10
        num_iters = 500
        learning_rate = 2e-2
        print(f"Performing the training loop with {num_epochs} epochs.\nBatch size is {batch_size}.\nLearning rate is set to {learning_rate}")
        train_loop(device, train_loader, net, simulation_steps,
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999)), 
                loss_fn =  SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2), 
                num_epochs = num_epochs, 
                num_iters = num_iters)
        torch.save(net.state_dict(), model_filename)
    else:
        print(f"Loading weights from {model_filename}")
        net.load_state_dict(torch.load(model_filename, weights_only=True))

    net.eval()
    test_accuracy = compute_accuracy(device, test_loader, net, simulation_steps)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%\n")