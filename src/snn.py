
import torch, snntorch
from snntorch import functional as SF, utils, surrogate
from snntorch import spikeplot as splt

import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

def get_neuron_model(neuron_model : snntorch.SpikingNeuron, **kwargs):
    """
    Get the neuron model to be used in the architecture. By default, we use Leaky neurons, but you can specify any neuron model from snntorch.

    The parameters for the neuron model can be specified as keyword arguments. For example, if you want to use Leaky neurons with a learnable beta parameter, you can specify learn_beta=True and beta=0.5 (or any other value) as keyword arguments.

    The parameters that can be specified as keyword arguments are:
        - spike_grad: surrogate gradient function to be used. Defaults to atan.
        - learn_alpha: Option to enable learnable alpha parameter for leaky neurons. Defaults to False
        - alpha: initial value for alpha parameter if learn_alpha is False. Defaults to 0.5
        - learn_beta: Option to enable learnable beta parameter for leaky neurons. Defaults to False
        - beta: initial value for beta parameter if learn_beta is False. Defaults to 0.5
        - threshold: initial value for threshold parameter if learn_threshold is False. Defaults to 1.0
        - learn_threshold: Option to enable learnable threshold parameter for leaky neurons. Defaults to False
        - reset_mechanism: mechanism for resetting the membrane potential after a spike. Can be "subtract" (default) or "zero".
        - reset_mechanism determines how the membrane potential is reset after a spike. In reset-by-subtraction, the threshold value is subtracted from the membrane potential :math:`mem` each time the threshold
        is met. In reset-to-zero, the membrane potential is reset to zero each time the threshold is met. If none, the membrane potential is not reset after a spike. Defaults to "subtract""

    """
    if neuron_model == snntorch.Leaky:
        return snntorch.Leaky(
            beta = kwargs.get("beta", 0.5),
            threshold = kwargs.get("threshold", 1.0),
            spike_grad = kwargs.get("spike_grad", surrogate.atan()),
            learn_beta = kwargs.get("learn_beta", False),
            learn_threshold = kwargs.get("learn_threshold", False),
            reset_mechanism = kwargs.get("reset_mechanism", "subtract"),
            init_hidden = kwargs.get("init_hidden", False),
            output = kwargs.get("output", False)
        )
    elif neuron_model == snntorch.Synaptic:
        return snntorch.Synaptic(
            alpha=kwargs.get("alpha", 0.5),
            beta=kwargs.get("beta", 0.5),
            threshold=kwargs.get("threshold", 1.0),
            spike_grad=kwargs.get("spike_grad", surrogate.atan()),
            learn_alpha=kwargs.get("learn_alpha", False),
            learn_beta=kwargs.get("learn_beta", False),
            learn_threshold=kwargs.get("learn_threshold", False),
            reset_mechanism=kwargs.get("reset_mechanism", "subtract"),
            init_hidden=kwargs.get("init_hidden", False),
            output=kwargs.get("output", False)
        )
    else:
        raise NotImplementedError(f"Neuron model {neuron_model} not implemented yet. Please implement the get_neuron_model function for this neuron model.")


def forward_pass(snn_model: torch.nn.Module, samples: torch.Tensor):
    mem_rec = []
    spk_rec = []
    utils.reset(snn_model)  # resets hidden states for all LIF neurons in net
    for step in range(samples.size(0)):  # data.size(0) = number of time steps
        spk_out, mem_out = snn_model(samples[step])
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
    return torch.stack(spk_rec), torch.stack(mem_rec)

def forward_pass_2nd(snn_model: torch.nn.Module, samples: torch.Tensor):
    spk_rec = []
    syn_rec = []
    mem_rec = []
    utils.reset(snn_model)  # resets hidden states for all LIF neurons in net
    for step in range(samples.size(0)):  # data.size(0) = number of time steps
        spk_out, syn_out, mem_out = snn_model(samples[step])
        spk_rec.append(spk_out)
        syn_rec.append(syn_out)
        mem_rec.append(mem_out)
    return torch.stack(spk_rec), torch.stack(syn_rec), torch.stack(mem_rec)

def compute_accuracy(
    device, dataloader : torch.utils.data.DataLoader,
    snn_model: torch.nn.Module,
    forward_pass_fn : callable = forward_pass,
    acc_fn : callable = SF.accuracy_rate):
    with torch.no_grad():
        total, correct = 0, 0
        snn_model.eval()
        dataloader = iter(dataloader)
        for data, targets in tqdm(dataloader, desc="Evaluating the model on the test set:", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
            data, targets = data.to(device), targets.to(device)
            spk_rec, _ = forward_pass_fn(snn_model, data)
            correct += acc_fn(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)
    return correct / total


def plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec, outfile = "spk_cur_mem_spk_plot.pdf"):
    # Generate Plots
    fig, ax = plt.subplots(4, figsize=(8,7), sharex=True,
                        gridspec_kw = {'height_ratios': [0.4, 1, 1, 0.4]})

    # Plot input current
    splt.raster(spk_in, ax[0], s=400, c="black", marker="|")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title("Synaptic Conductance-based Neuron Model With Input Spikes")
    ax[0].set_yticks([])

    # Plot membrane potential
    ax[1].plot(syn_rec.detach().numpy())
    ax[1].set_ylim([0, 0.5])
    ax[1].set_ylabel("Synaptic Current ($I_{syn}$)")
    plt.xlabel("Time step")

    # Plot membrane potential
    ax[2].plot(mem_rec.detach().numpy())
    ax[2].set_ylim([0, 1.5])
    ax[2].set_ylabel("Membrane Potential ($U_{mem}$)")
    ax[2].axhline(y=1, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk_rec, ax[3], s=400, c="black", marker="|")
    plt.ylabel("Output spikes")
    ax[3].set_yticks([])

    plt.savefig(outfile)

def plot_spk_mem_spk(spk_in, mem_rec, spk_rec, title, outfile = "spk_mem_spk_plot.pdf"):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,6), sharex=True,
                        gridspec_kw = {'height_ratios': [0.4, 1, 0.4]})

    # Plot input current
    splt.raster(spk_in, ax[0], s=400, c="black", marker="|")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)
    ax[0].set_yticks([])

    # Plot membrane potential
    ax[1].plot(mem_rec.detach())
    ax[1].set_ylim([0, 0.6])
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
    ax[1].axhline(y=0.5, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk_rec, ax[2], s=400, c="black", marker="|")
    ax[2].set_yticks([])
    ax[2].set_ylabel("Output Spikes")

    plt.savefig(outfile)