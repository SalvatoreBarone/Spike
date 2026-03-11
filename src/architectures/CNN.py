import torch.nn as nn,snntorch
from .GenericSNN import GenericSNN
from ..snn import get_neuron_model

class Network(GenericSNN):
    """
    The convolutional network architecture to be used is the one from:
    K. Stewart, G. Orchard, S. B. Shrestha, e E. Neftci, «Online Few-Shot Gesture Learning on a
    Neuromorphic Processor», IEEE J. Emerg. Sel. Topics Circuits Syst., vol. 10, fasc. 4, pp. 512–521,
    dic. 2020, doi: 10.1109/JETCAS.2020.3032058.
    """
    def __init__(self, input_depth : int, noutputs : int, neuron_model : snntorch.SpikingNeuron, **kwargs):
        super().__init__()
        self.model = self._get_model(input_depth, noutputs, neuron_model, **kwargs)
        
    def _get_model(self, input_depth : int, noutputs : int, neuron_model : snntorch.SpikingNeuron, **kwargs):
        return nn.Sequential(
            nn.MaxPool2d(4),  # downsample input size from 128x128 to 32x32
            nn.Conv2d(
                in_channels=input_depth, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.MaxPool2d(2),
            get_neuron_model(neuron_model, init_hidden=True, kwargs = kwargs),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            get_neuron_model(neuron_model, init_hidden=True, kwargs = kwargs),
            
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=512),
            get_neuron_model(neuron_model, init_hidden=True, kwargs = kwargs),
            
            nn.Linear(in_features=512, out_features=noutputs),
            get_neuron_model(neuron_model, init_hidden=True, output=True, kwargs = kwargs),
        )