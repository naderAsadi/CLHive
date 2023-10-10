from typing import Any, Callable, Dict, Tuple
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import register_dataset


# @register_dataset("tiny_imagenet")
class TinyImageNet(Dataset):
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self, root: str, image_transform: Callable, split: str = "train", **kwargs
    ):
        assert split in [
            "train",
            "val",
        ], f"`split` should be in [`train`, `val`], but {split} is entered."

        self.id_dict, self.filenames, self.cls_index = self._process_dataset(
            root=root, split=split
        )
        self.image_transform = image_transform

    @classmethod
    def from_config(
        cls,
        data_config: Dict[str, Any],
        split: str = "train",
        **kwargs,
    ) -> "TinyImageNet":
        if split == "train":
            image_transform = T.Compose(
                [
                    T.RandomResizedCrop(data_config.image_size),
                    T.RandomHorizontalFlip(p=0.5),
                    T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
                    T.ToTensor(),
                    T.Normalize(mean=TinyImageNet._MEAN, std=TinyImageNet._STD),
                ]
            )
        elif split == "test":
            split = "val"
            image_transform = T.Compose(
                [
                    T.Resize(data_config.image_size),
                    T.ToTensor(),
                    T.Normalize(mean=TinyImageNet._MEAN, std=TinyImageNet._STD),
                ]
            )
        else:
            raise ValueError(
                f"`split` should be in [`train`, `test`], but {split} is entered."
            )

        return cls(root=data_config.root, image_transform=image_transform, split=split)

    def _process_dataset(self, root: str, split: str):
        # Create a dictionary of class ids and class names
        id_dict = {}
        for i, line in enumerate(open(os.path.join(root, "wnids.txt"), "r")):
            id_dict[line.replace("\n", "")] = i

        # Create a list of all of the images
        if split == "train":
            filenames = glob.glob(os.path.join(root, split, "*/*/*.JPEG"))
            cls_index = -3
        else:
            filenames = glob.glob(os.path.join(root, split, "images/*.JPEG"))
            cls_index = -1
            # Process validation annotations file
            cls_dic = {}
            for i, line in enumerate(
                open(os.path.join(root, split, "val_annotations.txt"), "r")
            ):
                a = line.split("\t")
                img_id, cls_id = a[0], a[1]
                cls_dic[img_id] = id_dict[cls_id]

            id_dict = cls_dic

        return id_dict, filenames, cls_index

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        assert index >= 0 and index < len(
            self.filenames
        ), f"Provided index ({index}) is outside of dataset range."

        img_path = self.filenames[index]
        image = Image.open(img_path).convert("RGB")

        label = self.id_dict[img_path.split("/")[self.cls_index]]
        if self.image_transform:
            image = self.image_transform(image)

        return image, label
