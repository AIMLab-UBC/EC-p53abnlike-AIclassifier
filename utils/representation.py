import os
import sys
import random
from collections import OrderedDict

import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from utils.utils import print_title, print_title_, print_color
from models.model import VanillaModel
from utils.utils import save_dict, read_dict
from utils.dataloader import RepresentationDataset


class Representation(object):
    def __init__(self,
                 cfg: dict) -> None:
        self.cfg    = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.states = ['train', 'validation', 'test'] if not self.cfg['dataset']['use_external'] \
            else ['train', 'validation', 'test', 'external']
        self.states = [cfg['state']] if cfg['state'] is not None else self.states
        self.slide_id = cfg['slide_id']

        self.load_model()

    def load_model(self) -> None:
        """
        Loading the trained model in order to extract embedding of patches
        """
        state = torch.load(self.cfg['saved_model_location'], map_location=self.device)
        if self.cfg['method'] == 'Vanilla':
            model = VanillaModel(self.cfg['backbone'])
            state = state['model']
        else:
            raise NotImplementedError

        model.eval()
        self.model = model.to(self.device)
        missing_keys, error_keys = self.model.load_state_dict(state, strict=False)
        ## multiple GPU
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        ## check to make sure weights are loaded correctly
        if missing_keys:
            raise ValueError(f'model is not loaded correctly!')
        if self.cfg['method'] == 'Vanilla':
            for key in error_keys:
                if 'classifier' not in key:
                    raise ValueError('Only classifier should be among unexpected keys!')

    def calculate_representation(self) -> None:
        """
        Deriving patch embeddings
        """
        print_title('Deriving patch embeddings')
        print_title('Deriving patch embeddings')
        with torch.no_grad():
            for state in self.states:
                prefix = f'{state} representation: '
                dataloader = RepresentationDataset(self.cfg['dataset'], state, self.slide_id).run()
                for data in tqdm(dataloader, desc=prefix,
                        dynamic_ncols=True, leave=True, position=0):
                    data, slide_id, coords = data
                    data  = data.cuda() if torch.cuda.is_available() else data
                    representation = self.model(data)
                    self.save_representation(representation, slide_id, coords, state)
        # print_color('\nDone!', 'MAGENTA')
        print('\nDone!')

    def save_representation(self,
                            representation: torch.Tensor,
                            slide_id: tuple,
                            coords: list[torch.Tensor],
                            state: str) -> None:
        """
        Saving patch embeddings. Each slide will be saved as a dict with .pkl file

            slide_id.pkl : {'x1_y1': embeddings of the first patch,
                            'x2_y2': embeddings of the second patch,
                            .......
                            }

        Parameters
        ----------
        representation: torch.Tensor (N * d)
            a batch of derived embeddings
        slide_id: tuple (N)
            a tuple of slides' ids
        coords: list[torch.Tensor]
            the coordinates of the extracted patches
        state: str
            determining the slide belongs to which of train/validation/test
        """
        coords = [coords_.tolist() for coords_ in coords] # change tensor to list
        coords = np.array(coords).T.tolist() # change list of 2*n to n*2
        representation = list(representation.cpu().detach().numpy())
        for (repr_, id_, coord_) in zip(representation, slide_id, coords):
            dict_path = state
            if state == 'external' and self.cfg['dataset']['external_test_name'] != None:
                dict_path = f"external_{self.cfg['dataset']['external_test_name']}"
            path = os.path.join(self.cfg['representation_dir'], f"{dict_path}/{id_}.pkl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            dict_repr = read_dict(path) if os.path.isfile(path) else {}
            x, y = coord_
            dict_repr[f'{x}_{y}'] = repr_
            save_dict(dict_repr, path)

    def run(self) -> None:
        self.calculate_representation()
