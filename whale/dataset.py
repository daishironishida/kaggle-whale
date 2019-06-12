from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import random
from torch.utils.data import Dataset

from .transforms import tensor_transform
from .utils import ON_KAGGLE


DATA_ROOT = Path('../input/humpback-whale-identification' if ON_KAGGLE else './data')


class CustomDataset(Dataset):
    def initialize_epoch(self):
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TrainDataset(CustomDataset):
    def __init__(self, root:Path, df: pd.DataFrame,
                 image_transform: Callable, debug: bool=True):
        self.root = root
        self.df = df
        self.image_transform = image_transform
        self.debug = debug
        self.train_list = []

    # 一周したらペアのリストを作り直す
    def initialize_epoch(self):
        # different pairs
        while True:
            img_list = []
            remaining = list(self.df.index)
            for _, row in self.df.iterrows():
                found_pair = False
                for _ in range(len(remaining)):
                    poten_idx = random.randrange(len(remaining))
                    poten_image = remaining[poten_idx]
                    if row.Id != self.df.at[poten_image, 'Id']:
                        remaining.pop(poten_idx)
                        img_list.append(poten_image)
                        found_pair = True
                        break
                if not found_pair:
                    break
            if found_pair:
                break

        self.train_list = [(x, y, 0) for x, y in zip(img_list, self.df.index)]

        # matching pairs
        for whale_id in pd.unique(self.df.Id):
            images_list = list(self.df[self.df.Id == whale_id].index)
            self.train_list += [(x, y, 1) for x, y in derange(images_list)]

    def __len__(self):
        return 2 * len(self.df)

    def __getitem__(self, idx):

        if idx >= len(self.train_list):
            print("idx: ", idx)
            print("train list: ", len(self.train_list))

        img0_name, img1_name, label = self.train_list[idx]
        image0 = load_transform_image(
            img0_name, self.root, self.image_transform, debug=self.debug)
        image1 = load_transform_image(
            img1_name, self.root, self.image_transform, debug=self.debug)

        return image0, image1, label


class ValDataset(CustomDataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable):
        self.root = root
        self.df = df
        self.image_transform = image_transform
        self.num_items = self.df.Id.value_counts()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image0 = load_transform_image(item.name, self.root, self.image_transform)

        if self.num_items[item.Id] == 1:
            indices = list(range(len(self.df)))
            indices.remove(idx)
            image1 = load_transform_image(
                self.df.iloc[random.choice(indices)].name,
                self.root, self.image_transform)
            return image0, image1, 0

        else:
            if random.random() < 0.5:
                images_list = list(self.df[self.df.Id == item.Id].index)
                images_list.remove(item.name)
                image1 = load_transform_image(
                    random.choice(images_list), self.root, self.image_transform)
                return image0, image1, 1
            else:
                images_list = list(self.df[self.df.Id != item.Id].index)
                image1 = load_transform_image(
                    random.choice(images_list), self.root, self.image_transform)
                return image0, image1, 0


class TestDataset(CustomDataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable):
        self.root = root
        self.df = df
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = load_transform_image(item.index, self.root, self.image_transform)
        return image, item.Image


def derange(list_a):
    list_a = np.array(list_a)
    list_b = list_a.copy()
    while True:
        random.shuffle(list_b)
        if not np.any(list_a == list_b):
            break
    return list(zip(list_a, list_b))


def load_transform_image(
        item, root: Path, image_transform: Callable, debug: bool = False):
    image = load_image(item, root)
    image = image_transform(image)
    if debug:
        image.save('_debug.png')
    return tensor_transform(image)


def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / item))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
