import torch
import numpy as np
from skimage import transform
from typing import Union, Tuple, List

class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size: Union[int, Tuple]) -> None:
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample: Tuple[np.ndarray, int]) -> Tuple[np.ndarray, int]:
        image, label = sample[0], sample[1]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img, label


class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size: Union[int, Tuple]) -> None:
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample: Tuple[np.ndarray, int]) -> Tuple[np.ndarray, int]:
        image, label = sample[0], sample[1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image, label


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: Tuple[np.ndarray, int]) -> Tuple[torch.tensor, int]:
        image, label = sample[0], sample[1]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image), label


class Normalize:
    """
    Normalize the image either according to imagenet stats or to have a mean of 0 and std of 1
    """

    def __init__(self, use_imagenet_stats: bool = True) -> None:
        self.use_imagenet_stats = use_imagenet_stats
        if self.use_imagenet_stats:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

    def __call__(self, sample: Tuple[np.ndarray, int]) -> Tuple[np.ndarray, int]:
        image, label = sample[0], sample[1]

        if self.use_imagenet_stats:
            image[:][:][0] = (image[:][:][0] - self.mean[0]) / self.std[0]
            image[:][:][1] = (image[:][:][1] - self.mean[1]) / self.std[1]
            image[:][:][2] = (image[:][:][2] - self.mean[2]) / self.std[2]
        else:
            for channel in range(image.shape[2]):
                channel_mean = image[:][:][channel].mean()
                channel_std = image[:][:][channel].std()
                image[:][:][channel] = (image[:][:][channel] - channel_mean) / channel_std

        return image, label