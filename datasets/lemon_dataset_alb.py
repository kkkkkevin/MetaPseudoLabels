import math
from albumentations.augmentations.transforms import VerticalFlip
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

# Albumenatations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_lemon_datasets(args):

    pretrain_mean = (0.485, 0.456, 0.406)
    pretrain_std = (0.229, 0.224, 0.225)

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
        transform=get_transforms_labeled(
            args.resize,
            mean=pretrain_mean,
            std=pretrain_std))
    train_unlabeled_dataset = LemonDataset(
        train_unlabeled_idxs,
        train_i,
        train_l,
        transform=TransformMPL(
            args.img_size,
            args.resize,
            mean=pretrain_mean,
            std=pretrain_std))
    test_dataset = LemonDataset(
        range(len(test_i)),
        test_i,
        test_l,
        transform=get_transforms_val(mean=pretrain_mean, std=pretrain_std))

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
        assert len(idx) >= label_per_class, \
            "class_id:{i} of num is less than '(num_labels // num_classes)'"
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
        # image
        img = cv2.imread(self.img_paths[index], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        # img /= 255.0
        # label
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TransformMPL(object):
    def __init__(
            self,
            img_size: tuple,
            resize: int,
            mean: tuple,
            std: tuple
    ) -> None:
        self.ori = get_transforms_labeled(resize, mean, std)
        self.aug = get_transforms_unlabeled(img_size, resize, mean, std)

    def __call__(self, image):
        ori = self.ori(image=image)["image"]
        aug = self.aug(image=image)["image"]
        return {"image": (ori, aug)}


def get_transforms_labeled(resize: int, mean: tuple, std: tuple) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=180,
            interpolation=1,
            border_mode=4,
            p=0.5),
        A.RandomCrop(
            height=resize,
            width=resize,
            p=1),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255,
            p=1),
        ToTensorV2(p=1)
    ])


def get_transforms_unlabeled(
        img_size: tuple,  # [height, width]
        resize: int,
        mean: tuple,
        std: tuple) -> A.Compose:
    return A.Compose([
        A.RandomSizedCrop(
            min_max_height=[480, img_size[0]],
            height=resize,
            width=resize,
            w2h_ratio=1.,
            p=1),
        # Color
        A.OneOf([
            A.Equalize(
                mode='cv',
                by_channels=True,
                mask=None,
                mask_params=(),
                p=0.5),
            A.ToGray(p=0.5),
        ], p=0.2),
        # Luminance, Contrast
        A.OneOf([
            A.RandomContrast(
                limit=(-0.2, 0.2),
                p=0.5),
            A.InvertImg(
                p=0.5),
            A.RandomBrightness(
                limit=0.2,
                p=0.5),
            A.Posterize(
                num_bits=4,
                p=0.5),
            A.Solarize(
                threshold=128,
                p=0.5),
        ], p=0.5),
        # Sharp, Blur
        A.OneOf([
            A.IAASharpen(
                alpha=(0.2, 0.5),
                lightness=(0.5, 1.0),
                p=0.5),
            A.Blur(
                blur_limit=(2, 6),
                p=0.5),
        ], p=0.2),
        # Rotate
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.Transpose(p=1),
            A.RandomRotate90(p=1),
            A.Rotate(
                limit=180,
                interpolation=1,
                border_mode=4,
                p=0.5),
        ], p=0.5),
        # Dropout
        A.Cutout(
            num_holes=8,
            max_h_size=8,
            max_w_size=8,
            fill_value=0,
            p=0.5),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255,
            p=1),
        ToTensorV2(p=1)
    ])


def get_transforms_val(mean: tuple, std: tuple) -> A.Compose:
    return A.Compose([
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255,
            p=1),
        ToTensorV2(p=1)
    ])
