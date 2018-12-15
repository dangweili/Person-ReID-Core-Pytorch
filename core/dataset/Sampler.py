import torch
from torch.utils.data.sampler import Sampler
import copy
import numpy as np
import random

class RandomIdentitySampler(Sampler):
    """
    generate the batch index for data loader
    """
    def __init__(
        self, 
        dataset,
        num_instances=1):
        """
        dataset: the ReIDDataset
        num_instances: for each person, how many images are sampled
        """
        # the dataset contains three lists, including image, label, camera
        self.dataset = dataset
        self.num_instances = num_instances
        if num_instances == 1:
            print 'The num_instances in RandomIdentitySampler should be larger than 1.'
            raise ValueError
        self.create_dataset()

    def create_dataset(self):
        """
        Create the dict for sampling
        """
        data = dict()
        for idx, pid in enumerate(self.dataset.label):
            if not data.has_key( pid ):
                data[ pid ] = [] 
            data[pid].append( idx )
        self.pids = data.keys() 
        self.data = data
        # the number of identities
        self.num_samples = len(self.pids)
    
    def __iter__(self):
        # shuffle the person identities order
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            tmp_rec = copy.deepcopy(self.data[pid])
            # sample self.num_instances samples for pid from c cameras
            N_left = self.num_instances - len(tmp_rec)
            if N_left > 0:
                ret.extend( np.random.choice(tmp_rec, size=self.num_instances, replace=True))
            else:
                ret.extend( np.random.choice(tmp_rec, size=self.num_instances, replace=False))
        return iter(ret)            

    def __len__(self):
        return self.num_samples * self.num_instances

class RandomIdentitySamplerXC(Sampler):
    """
    generate the batch index for data loader
    """
    def __init__(
        self, 
        dataset,
        num_instances=1):
        """
        dataset: the ReIDDataset
        num_instances: for each person, how many images are sampled
        """
        # the dataset contains three lists, including image, label, camera
        self.dataset = dataset
        self.num_instances = num_instances
        if num_instances == 1:
            print 'The num_instances in RandomIdentitySamplerXC should be larger than 1.'
            raise ValueError
        self.create_dataset()

    def create_dataset(self):
        """
        Create the dict for sampling
        """
        data = dict()
        for idx, pid in enumerate(self.dataset.label):
            if not data.has_key( pid ):
                data[ pid ] = dict()
            cam = self.dataset.camera[idx]
            if not data[pid].has_key( cam ):
                data[pid][cam] = []
            data[pid][cam].append( idx )
        self.pids = data.keys() 
        self.data = data
        # the number of identities
        self.num_samples = len(self.pids)
    
    def __iter__(self):
        # shuffle the person identities order
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            tmp_ret = []
            tmp_rec = copy.deepcopy(self.data[pid])
            # sample self.num_instances samples for pid from c cameras
            while True:
                cam_list = tmp_rec.keys()
                if len(cam_list) == 0:
                    break
                # sample different cameras at different epochs
                random.shuffle(cam_list)
                for cam in cam_list:
                    # sample one instance for each time at each camera
                    tmp_ret.extend( np.random.choice(tmp_rec[cam], size=1, replace=False) )
                    tmp_rec[cam].remove(tmp_ret[-1])

                    if len(tmp_rec[cam]) == 0:
                        del tmp_rec[cam]
                    # break the for if the images are ok
                    if len(tmp_ret) == self.num_instances:
                        break
                # break the while if the images are ok
                if len(tmp_ret) == self.num_instances:
                    break
            N_left = self.num_instances - len(tmp_ret)
            if N_left > 0:
                if len(tmp_ret) >= N_left:
                    tmp_ret.extend( np.random.choice(tmp_ret, size=N_left, replace=False) )
                else:
                    tmp_ret.extend( np.random.choice(tmp_ret, size=N_left, replace=True) )
            ret.extend( tmp_ret )
        return iter(ret)            

    def __len__(self):
        return self.num_samples * self.num_instances

