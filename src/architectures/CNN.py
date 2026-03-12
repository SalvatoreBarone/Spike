import torch.nn as nn,snntorch
from .GenericSNN import GenericSNN
from ..snn import get_neuron_model

class MP4_16C5_MP2_32C3_MP2_2048FC512_512FCnclasses(GenericSNN):
    """
    The convolutional network architecture to be used is the one from:
    K. Stewart, G. Orchard, S. B. Shrestha, e E. Neftci, «Online Few-Shot Gesture Learning on a
    Neuromorphic Processor», IEEE J. Emerg. Sel. Topics Circuits Syst., vol. 10, fasc. 4, pp. 512–521,
    dic. 2020, doi: 10.1109/JETCAS.2020.3032058.
    
    The network is a MP4-16C5-MP2-32C3-MP2-2048FC512-512FCnclasses architecture, where
        - MP is max pooling,
        - C is convolutional layer, and
        - FC is fully connected layer.
    The first convolutional layer has 16 filters of size 5x5, the second convolutional layer has 32
    filters of size 3x3, and the fully connected layers have 2048 and 512 neurons respectively.
    The output layer has nclasses neurons, where nclasses is the number of classes in the dataset.
    The max pooling layers have a kernel size of 4x4 and 2x2 respectively.
    The activation function used is the spiking neuron model specified in the constructor.
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