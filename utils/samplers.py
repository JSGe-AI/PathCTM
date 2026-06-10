import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import math
import itertools
import numpy as np

class FastRandomDistributedSampler(Sampler[int]):

    def __init__(self, dataset, num_replicas=None, rank=None, seed=0, epoch_steps=None):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("d")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("ed")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.dataset_len = len(self.dataset)


        if epoch_steps is None:

            self.num_samples_per_epoch = math.ceil(self.dataset_len / self.num_replicas)
        else:

            self.num_samples_per_epoch = epoch_steps

        if not isinstance(self.num_samples_per_epoch, int) or self.num_samples_per_epoch <= 0:
            raise ValueError("epoch_steps must be a positive integer")

    def _infinite_indices(self):
  
        g = torch.Generator()
      
        current_seed = self.seed + self.epoch * self.num_replicas + self.rank
        g.manual_seed(current_seed)
        while True:
            yield torch.randint(low=0, high=self.dataset_len, size=(1,), generator=g).item()

    def __iter__(self):
      
   
        return itertools.islice(self._infinite_indices(), self.num_samples_per_epoch)

    def __len__(self):
        
        return self.num_samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
   
        self.epoch = epoch

class QAMNISTSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            if self.dataset.num_images_range[0] == self.dataset.num_images_range[1]:
                batch_num_digits = self.dataset.num_images_range[0]
            else:
                batch_num_digits = np.random.randint(self.dataset.num_images_range[0], self.dataset.num_images_range[1])

            if self.dataset.num_operations_range[0] == self.dataset.num_operations_range[1]:
                batch_num_operations = self.dataset.num_operations_range[0]
            else:
                batch_num_operations = np.random.randint(self.dataset.num_operations_range[0], self.dataset.num_operations_range[1])

            self.dataset.set_num_digits(batch_num_digits)
            self.dataset.set_num_operations(batch_num_operations)
            
            yield batch_indices

    def __len__(self):
        return self.num_samples // self.batch_size