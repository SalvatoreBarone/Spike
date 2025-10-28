import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn

from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
from torch.utils.data import DataLoader

import os

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
  for _ in range(simulation_steps):
      spk_out, mem_out = snn_model(samples)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)
  return torch.stack(spk_rec), torch.stack(mem_rec)

def compute_accuracy(device, dataloader, snn_model, simulation_steps, population_code = False, num_classes = None):
  with torch.no_grad():
    total = 0
    acc = 0
    snn_model.eval()
    dataloader = iter(dataloader)
    for data, targets in dataloader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(snn_model, simulation_steps, data)
      if population_code:
        acc += SF.accuracy_rate(spk_rec, targets, population_code = population_code, num_classes = num_classes) * spk_rec.size(1)
      else:
        acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)    
      total += spk_rec.size(1)
  return acc/total

# Torch Variables
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")

# Training Parameters
batch_size = 128
num_epochs = 5
learning_rate = 1e-3

# snnTorch Parameters
simulation_steps = 50
spike_grad = surrogate.fast_sigmoid()
decay_rate = 0.9  

# Output file Name for learned weights
filename = "tutorial8.pth"

# Instead of 10 output neurons corresponding to 10 output classes, 
# we will use 500 output neurons. This means each output class has
# 50 neurons randomly assigned to it.
pop_outputs = 500

## We use a 12C5-MP2-64C5-MP2-1024FC-PopSize
snn_model = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, pop_outputs), ## Instead of 10 output neurons corresponding to 10 output classes
                                                    ## we will use multiple output neurons. Each class has a number of
                                                    ## neurons randomly assigned to it
                    snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

print(f"Network model:\n{snn_model}")

# Loss function and optimizer
optimizer = torch.optim.Adam(snn_model.parameters(), lr = learning_rate, betas=(0.9, 0.999))
# We need to specify the use of population code, as well as the number of classes
#! Doesn't work with ce_rate_loss()...
loss_function = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=10)

## Setting up the static MNIST
mnist_train, mnist_test = get_MNIST()

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

if not os.path.exists(filename):
    print(f"Performing the training loop with {num_epochs} epochs.\nBatch size is {batch_size}.\nLearning rate is set to {learning_rate}")
    # Outer training loop
    avg_losses = []
    iteration_counter = 0
    for epoch in range(num_epochs):
      avg_loss = backprop.BPTT(snn_model, train_loader, num_steps=simulation_steps,
                          optimizer=optimizer, criterion=loss_function, time_var=False, device=device)
      avg_losses.append(avg_loss.item())
      print(f"\n\nEpoch: {epoch}\n\tLoss: {avg_loss:.3f}\n\tTest set accuracy: {compute_accuracy(device, test_loader, snn_model, simulation_steps, population_code=True, num_classes=10)*100:.3f}% ")
      

    torch.save(snn_model.state_dict(), filename)
    plt.plot(range(len(avg_losses)), avg_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title("Average Training Loss vs Epochs")
    plt.savefig("tutorial8_loss.pdf")
else:
    print(f"Loading weights from {filename}")
    snn_model.load_state_dict(torch.load(filename, weights_only=True))

test_acc = 0
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)
with torch.no_grad():
  snn_model.eval()
  test_acc = compute_accuracy(device, test_loader, snn_model, simulation_steps, population_code=True, num_classes=10)

print(f"Accuracy on test set images: {test_acc*100:.2f}")
