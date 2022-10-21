import os
import pickle
import enum

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from utils.utils import print_color
from augmentation.ToTensor import ToTensor


class FeatDataset(Dataset):
    def __init__(self,
                 x_set: list[tuple],
                 resize: int = None,
                 method: str = 'representation',
                 slide_id: int = None,
                 state: str = None) -> None:
        self.x_set = x_set
        self.resize = resize
        self.method = method
        self.transform = self.get_transform()
        self.create_bag(x_set, slide_id, state)
        self.length = len(self.x_set)

    def get_transform(self) -> transforms:
        transforms_array = []

        if self.resize is not None:
            transforms_array.append(transforms.Resize(self.resize))

        transforms_array.append(transforms.ToTensor())
        transforms_array.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
        transforms_ = transforms.Compose(transforms_array)
        return transforms_

    def __len__(self) -> int:
        return self.length

    def create_bag(self,
                   x_set: list[str],
                   nb_id: int,
                   state: str = None) -> None:
        bag_x_set = {}
        for data in x_set:
            path, id = data
            if id not in bag_x_set:
                bag_x_set[id] = []
            bag_x_set[id].append([path, id])
        keys = list(bag_x_set.keys())

        if nb_id is None:
            print(f'{state.capitalize()}: Calculating feature embeddings for {len(keys)} slides ({len(self.x_set)} patches).')
            return

        if nb_id - 1 >= len(keys):
            raise ValueError(f'{nb_id} is bigger than the number of slides ({len(keys)}).')

        self.x_set = bag_x_set[keys[nb_id-1]]
        print(f'{state.capitalize()}: Calculating feature embeddings for the {keys[nb_id-1]} slide ({len(self.x_set)} patches).')

    def __getitem__(self,
                    idx: int) -> torch.Tensor | int | tuple:
        path = self.x_set[idx][0]
        x = Image.open(path).convert('RGB')
        x = self.transform(x)
        slide_id  = self.x_set[idx][1]
        (x_,  y_) = os.path.splitext(os.path.basename(path))[0].split('_')
        (x_,  y_) = (int(x_),  int(y_))
        return x, slide_id, (x_,y_)

class BagDataset(Dataset):
    def __init__(self, x_set, y_set, repr_location, state, CategoryEnum, training_set=False, external_test_name=None):

        """
        Args:
            x_set (string): List of paths to images
            y_set (int): Labels of each image in x_set
            idx_set (dict of lists): list of index
            training_set: Whether training dataset or not
        """
        self.create_bags(x_set, y_set, repr_location, state, external_test_name)
        self.training_set = training_set
        self.transform = self.get_transform()
        self.classes_(CategoryEnum, y_set)
        self.ratio_()

    def create_bags(self, x_set, y_set, location, state, external_test_name=None):
        self.slide_rep = {}
        dict_path = state
        if state == 'external' and external_test_name != None:
            dict_path = f"external_{external_test_name}"
        for (data, label) in zip(x_set,y_set):
            path, id = data
            x,  y = os.path.splitext(os.path.basename(path))[0].split('_')
            if id not in self.slide_rep:
                self.slide_rep[id] = {}
                self.slide_rep[id]['gt'] = label
                self.slide_rep[id]['coordinates'] = []
                self.slide_rep[id]['representation'] = f'{location}/{dict_path}/{id}.pkl'
            self.slide_rep[id]['coordinates'].append((x,y))
        self.keys = list(self.slide_rep.keys())

    def get_transform(self) -> transforms:
        transforms_ = transforms.Compose([ToTensor()])
        return transforms_

    def classes_(self,
                 CategoryEnum: enum.Enum,
                 y_set:list) -> None:
        self.classes = []
        for y_ in set(y_set):
            self.classes.append(CategoryEnum(y_).name)

    def ratio_(self) -> None:
        '''
        Find the ratio of each class compare to others
        useful for balancing
        it should be 1-real_ratio
        '''
        y_set = [self.slide_rep[id]['gt'] for id in self.keys]
        _, ratio = np.unique(y_set, return_counts=True)
        ratio = 1 / (ratio / len(y_set))
        self.ratio = [ratio_/sum(ratio) for ratio_ in ratio]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self,
                    idx: int) -> torch.Tensor | torch.Tensor | list | list:
        id = self.keys[idx]
        x_dict = load_dict(self.slide_rep[id]['representation'])
        x = [x_dict[f'{x}_{y}'] for (x,y) in self.slide_rep[id]['coordinates']]
        x = np.stack(x, axis=0)
        y = self.slide_rep[id]['gt']
        coord = self.slide_rep[id]['coordinates']
        # x = self.transform(x)
        return torch.tensor(x), torch.tensor(y), id, coord

def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
