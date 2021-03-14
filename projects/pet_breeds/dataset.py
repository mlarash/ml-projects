from fastai.vision import untar_data, URLs
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path, PosixPath
from PIL import Image
from skimage import io
import numpy as np
import re

from transforms import ToTensor

from typing import Tuple, List, Dict


class PetDataset:
    
    def __init__(self, transform=None) -> None:
        self.path_img = self.load_data()
        self.paths_to_labels, self.breed_to_int = self.build_dataset()
        self.transform = transform

    def build_dataset(self) -> Tuple[List[Tuple[PosixPath, int]], Dict[str, int]]:
        """
        Parses the image names and returns two items:
            1. paths_to_labels, a list of Tuple(Path, label)
            2. int_to_label, a dict mapping integer labels to their corresponding breed
        """
        pet_breed_idx = 0
        breed_to_int = {} 
        paths_to_labels = []

        for path in self.path_img.ls():
            try:
                image = io.imread(path)
                if image.shape[2] != 3 or len(image.shape) != 3: 
                    continue
            except:
                continue
            # get breed from regular expression of filename
            filename_regexpr = r'/([^/]+)_\d+.jpg$'
            try:
                breed = re.findall(filename_regexpr, str(path))[0].lower()
            except:
                continue
            if breed in breed_to_int:
                paths_to_labels.append((path, breed_to_int[breed]))
            else:
                breed_to_int[breed] = pet_breed_idx
                paths_to_labels.append((path, breed_to_int[breed]))
                pet_breed_idx += 1

        return paths_to_labels, breed_to_int

    @staticmethod
    def load_data() -> Path:
        """
        Returns tuple of path to annotations and path to images
        """
        # path = untar_data(URLs.PETS, dest="../data")
        path = Path('../../data/oxford-iiit-pet')
        path_img = path/'images'  # image folder contains jpeg images
        
        return path_img

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        """
        Process image at specified path and return Tuple of (tensor, label)
        """
        img_path, label = self.paths_to_labels[i]
        image = io.imread(img_path)

        # process image and turn into tensor
        sample = (image, label)
        if self.transform is not None:
            image, label = self.transform(sample)

        to_tensor = ToTensor()
        image, label = to_tensor((image, label))

        return image, label
        
    def __len__(self) -> int:
        return len(self.paths_to_labels)
