from .Dataset import Dataset
import tonic, tonic.transforms as transforms
from torchvision.transforms import v2
from tonic import MemoryCachedDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class DVSGestureTimeWindow(Dataset):
    def __init__(self, batch_size : int = 32, filter_time : int = 10000, time_window_us :  int = 20000, rank : int = 0, sampler = DistributedSampler):
        super().__init__()
        self.batch_size = batch_size
        self.filter_time = filter_time
        self.time_window_us = time_window_us
        self.rank = rank
        self.sampler = sampler
        self._get_dataset()        

    def _get_dataset(self):
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
        data_path = "data/"
        train_transform = transforms.Compose(
            [
                transforms.Denoise(filter_time = self.filter_time),
                transforms.ToFrame(
                    sensor_size=tonic.datasets.DVSGesture.sensor_size,
                    time_window=self.time_window_us,
                ),
                v2.RandomRotation(10),
                
            ]
        )
        trainset = tonic.datasets.DVSGesture(
            save_to=data_path, transform=train_transform, train=True
        )
        test_transform = transforms.Compose(
            [
                transforms.Denoise(filter_time = self.filter_time),
                transforms.ToFrame(
                    sensor_size=tonic.datasets.DVSGesture.sensor_size,
                    time_window=self.time_window_us,
                ),
            ]
        )
        testset = tonic.datasets.DVSGesture(
            save_to=data_path, transform=test_transform, train=False
        )

        """
        Since the original data is stored in a format that is slow to read, Tonic will convert and cache the transformed data for
        faster loading in subsequent runs. That means that once files are loaded from the original dataset, they are written to
        the disk. 
        If you have a large amount of RAM available, you can speed up dataloading further by caching to main memory instead of to disk.

        """   
        self.train_set = MemoryCachedDataset(trainset)
        self.test_set =  MemoryCachedDataset(testset)
        
        """
        Because event recordings have different lengths, we are going to provide a collation function
        tonic.collation.PadTensors() that will pad out shorter recordings to ensure all samples in a batch have the same dimensions.

        Note: we don't need to specify the simulation time steps for the SNN as a parameter here.
        The number of time steps will be set to that of the longest recording in the mini-batch, and all other samples will be
        padded with zeros to match it.

        The structure of the input data must be [num_steps x batch_size x (input dimensions)] e.g. (310 x 128 x 2 x 32 x 32). 
        Therefore, the batch_first=False is of utter importance
        """
        self.train_loader = DataLoader(
            self.train_set,
            batch_size = self.batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            shuffle=False,
            sampler = self.sampler(self.train_set) if self.sampler is not None else None,
        )
        self.test_loader = DataLoader(
            self.test_set, collate_fn=tonic.collation.PadTensors(batch_first=False)
        )

        data_it, _ = next(iter(self.train_loader))
        self.ishape = data_it.size()
        self.nclasses = len(tonic.datasets.DVSGesture.classes)
        self.classes = tonic.datasets.DVSGesture.classes
        

class DVSGestureTimeBins(Dataset):
    def __init__(self, batch_size : int = 32, filter_time : int = 10000, n_time_bins : int = 32, rank : int = 0, sampler = DistributedSampler):
        super().__init__()
        self.width = 32
        self.height = 32
        self.depth = 2
        self.batch_size = batch_size
        self.filter_time = filter_time
        self.n_time_bins = n_time_bins
        self.rank = rank
        self.sampler = sampler
        self._get_dataset()
        
        
    def _get_dataset(self):
        """
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
        data_path = "data/"
        train_transform = transforms.Compose(
            [
                transforms.Denoise(filter_time = self.filter_time),      # removes outlier events with inactive surrounding 
                                                                         # pixels for filter_time microseconds. Smaller
                                                                         # filter_time will filter more events.
                # tonic.transforms.Downsample(
                #     sensor_size=tonic.datasets.DVSGesture.sensor_size, 
                #     target_size=(self.width, self.height)),              # downsampling image
                transforms.ToFrame(
                    sensor_size = tonic.datasets.DVSGesture.sensor_size, #(self.width, self.height, self.depth),
                    n_time_bins = self.n_time_bins,                      # n_frames frames per trail
                ),
                v2.RandomRotation(10),
                
            ]
        )
               
        trainset = tonic.datasets.DVSGesture(
            save_to=data_path, transform=train_transform, train=True
        )
        
        test_transform = transforms.Compose(
            [
                transforms.Denoise(filter_time = self.filter_time),      # removes outlier events with inactive surrounding 
                                                                         # pixels for filter_time microseconds. Smaller
                                                                         # filter_time will filter more events.
                # tonic.transforms.Downsample(
                #     sensor_size=tonic.datasets.DVSGesture.sensor_size, 
                #     target_size=(self.width, self.height)),              # downsampling image
                transforms.ToFrame(
                    sensor_size = tonic.datasets.DVSGesture.sensor_size, #(self.width, self.height, self.depth),
                    n_time_bins = self.n_time_bins,                      # n_frames frames per trail
                ),
            ]
        )
        testset = tonic.datasets.DVSGesture(
            save_to=data_path, transform=test_transform, train=False
        )
        
        """
        Since the original data is stored in a format that is slow to read, Tonic will convert and cache the transformed data for
        faster loading in subsequent runs. That means that once files are loaded from the original dataset, they are written to
        the disk. 
        If you have a large amount of RAM available, you can speed up dataloading further by caching to main memory instead of to disk.

        """   
        self.train_set = MemoryCachedDataset(trainset)
        self.test_set =  MemoryCachedDataset(testset)
        
        """
        Because event recordings have different lengths, we are going to provide a collation function
        tonic.collation.PadTensors() that will pad out shorter recordings to ensure all samples in a batch have the same dimensions.

        Note: we don't need to specify the simulation time steps for the SNN as a parameter here.
        The number of time steps will be set to that of the longest recording in the mini-batch, and all other samples will be
        padded with zeros to match it.

        The structure of the input data must be [num_steps x batch_size x (input dimensions)] e.g. (310 x 128 x 2 x 32 x 32). 
        Therefore, the batch_first=False is of utter importance
        """
        self.train_loader = DataLoader(
            self.train_set,
            batch_size = self.batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            shuffle=False,
            sampler=self.sampler(self.train_set) if self.sampler is not None else None,
        )
        self.test_loader = DataLoader(
            self.test_set, 
            collate_fn=tonic.collation.PadTensors(batch_first=False)
        )

        data_it, _ = next(iter(self.train_loader))
        self.ishape = data_it.size()
        self.nclasses = len(tonic.datasets.DVSGesture.classes)
        self.classes = tonic.datasets.DVSGesture.classes
        
        