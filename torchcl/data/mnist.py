from typing import Callable, Dict, Optional, Tuple, Union

from torchvision.datasets.cifar import MNIST

from torchcl.data import register_dataset
from torchcl.data.base import BaseDataset
from torchcl.data.transforms import get_transform, BaseTransform


@register_dataset("seq_mnist")
class MNISTDataset(MNIST):

    def __init__(
        self,
        root: str, 
        transform: Optional[Union[BaseTransform, Callable]],
        train: bool,
        download: bool = True
    ) -> None:

        dataset = MNIST(root, train=train, download=download)
        if transform is None:
            transform = BaseTransform()
        
        super().__init__(
            dataset=dataset, transform=transform
        )
    
    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        original_img = self.not_aug_transform(img.copy())

        img = self.transform(img)

        return img, target