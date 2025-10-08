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
  for step in range(simulation_steps):
      spk_out, mem_out = snn_model(samples)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)
  return torch.stack(spk_rec), torch.stack(mem_rec)

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

# Torch Variables
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")


## Defining the Network and neuron
spike_grad = surrogate.fast_sigmoid(slope=25)
decay_rate = 0.5
snn_model = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    snn.Leaky(beta=decay_rate, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

print(f"Network model:\n{snn_model}")

# SNN time steps
simulation_steps = 50

filename = "tutorial6.pth"

## Setting up the stati MNIST
mnist_train, mnist_test = get_MNIST()

# Training Parameters
batch_size = 128
num_epochs = 1
learning_rate = 1e-2

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

if not os.path.exists(filename):
    loss_function = SF.ce_rate_loss()
    optimizer = torch.optim.Adam(snn_model.parameters(), lr = learning_rate, betas=(0.9, 0.999))
    loss_history = []
    test_acc_history = []
    iteration_counter = 0
    print(f"Performing the training loop with {num_epochs} epochs.\nBatch size is {batch_size}.\nLearning rate is set to {learning_rate}")

    # Outer training loop
    for epoch in range(num_epochs):
        # Training loop
        for data, targets in iter(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            # forward pass
            snn_model.train()
            spk_rec, _ = forward_pass(snn_model, simulation_steps, data)
            # initialize the loss & sum over time
            loss_val = loss_function(spk_rec, targets)
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            # Store loss history for future plotting
            loss_history.append(loss_val.item())
            # Test set
            if iteration_counter % 50 == 0:
                with torch.no_grad():
                    snn_model.eval()
                    # Test set forward pass
                    test_acc = compute_accuracy(device, test_loader, snn_model, simulation_steps)
                    print(f"Iteration {iteration_counter}, Test Acc: {test_acc * 100:.2f}%\n")
                    test_acc_history.append(test_acc.item())
            iteration_counter += 1
    torch.save(snn_model.state_dict(), filename)
else:
    print(f"Loading weights from {filename}")
    snn_model.load_state_dict(torch.load(filename, weights_only=True))

test_acc = 0
# drop_last switched to False to keep all samples
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
  snn_model.eval()
  test_acc = compute_accuracy(device, test_loader, snn_model, simulation_steps)

print(f"Accuracy on test set images: {test_acc*100:.2f}")
