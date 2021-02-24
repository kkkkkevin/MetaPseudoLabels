import math
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop, ToTensor, Normalize
from utils.augmentation import RandAugment

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


def get_lemon_datasets(args):
    # divide data into train, test
    df_label = pd.read_csv(args.train_label_file_path)
    img_paths = df_label['file_name'].values
    labels = df_label['label'].values

    train_i, test_i, train_l, test_l = train_test_split(
        img_paths, labels,
        test_size=102,
        shuffle=True,
        random_state=args.seed,
        stratify=labels)

    # divide train into label, unlabel
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        train_l,
        args.num_labeled,
        args.num_classes,
        args.expand_labels,
        args.batch_size,
        args.eval_step)

    # dataset
    train_labeled_dataset = LemonDataset(
        train_labeled_idxs,
        train_i,
        train_l,
        transform=get_transforms_labeled(args.resize))
    train_unlabeled_dataset = LemonDataset(
        train_unlabeled_idxs,
        train_i,
        train_l,
        transform=TransformMPL(
            args,
            args.randaug,
            args.resize,
            mean=cifar10_mean,
            std=cifar10_std))
    test_dataset = LemonDataset(
        range(len(test_i)),
        test_i,
        test_l,
        transform=get_transforms_val())

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(
        labels,
        num_labeled: int,
        num_classes: int,
        expand_labels: bool,
        batch_size: int,
        eval_step: int):
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []

    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)

    assert len(labeled_idx) == num_labeled

    if expand_labels or num_labeled < batch_size:
        num_expand_x = math.ceil(batch_size * eval_step / num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)

    return labeled_idx, unlabeled_idx


class LemonDataset(Dataset):
    def __init__(
            self,
            indexs,
            img_paths,
            targets,
            transform=None,
            target_transform=None) -> None:
        self.indexs = indexs

        if indexs is not None:
            self.img_paths = img_paths[indexs]
            self.targets = targets[indexs]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.indexs)

    def __getitem__(self, index: int):
        img = Image.open(self.img_paths[index]).convert('RGB')
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TransformMPL(object):
    def __init__(
            self,
            args,
            randaug: tuple,
            resize: int,
            mean: tuple,
            std: tuple
    ) -> None:
        n, m = randaug
        self.ori = Compose([
            RandomHorizontalFlip(),
            RandomCrop(
                size=resize,
                padding=int(resize * 0.125),
                padding_mode='reflect')
        ])

        self.aug = Compose([
            RandomHorizontalFlip(),
            RandomCrop(
                size=resize,
                padding=int(resize * 0.125),
                padding_mode='reflect'),
            RandAugment(args, n=n, m=m)
        ])

        self.normalize = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


def get_transforms_labeled(resize: int) -> Compose:
    return Compose([
        RandomHorizontalFlip(),
        RandomCrop(
            size=resize,
            padding=int(resize * 0.125),
            padding_mode='reflect'
        ),
        ToTensor(),
        Normalize(mean=cifar10_mean, std=cifar10_std)
    ])


def get_transforms_val() -> Compose:
    return Compose([
        ToTensor(),
        Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
