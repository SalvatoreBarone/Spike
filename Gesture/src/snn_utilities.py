import torch
from snntorch import functional as SF
from snntorch import utils

def forward_pass(snn_model, samples):
    mem_rec = []
    spk_rec = []
    utils.reset(snn_model)  # resets hidden states for all LIF neurons in net
    for step in range(samples.size(0)):
        spk_out, mem_out = snn_model(samples[step])
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
    return torch.stack(spk_rec), torch.stack(mem_rec)

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