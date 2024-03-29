# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageDraw


logger = logging.getLogger(__name__)

PARAMETER_MAX = 10
RESAMPLE_MODE = None


def AutoContrast(img, **kwarg):
    return ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, **kwarg):
    """ UnUsed
    """
    if v == 0:
        return img
    v = _float_parameter(v, max_v)
    v = int(v * min(img.size))
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def CutoutConst(img, v, max_v, **kwarg):
    v = _int_parameter(v, max_v)
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return ImageOps.equalize(img)


def Identity(img, **kwarg):
    """ UnUsed
    """
    return img


def Invert(img, **kwarg):
    return ImageOps.invert(img)


def Posterize(img, v, max_v, bias, **kwarg):
    v = _int_parameter(v, max_v) + bias
    return ImageOps.posterize(img, v)


def Rotate(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.transform(
        img.size, Image.AFFINE, (1, v, 0, 0, 1, 0), RESAMPLE_MODE)


def ShearY(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, v, 1, 0), RESAMPLE_MODE)


def Solarize(img, v, max_v, **kwarg):
    v = _int_parameter(v, max_v)
    return ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, threshold=128, **kwarg):
    v = _int_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(
        img.size, Image.AFFINE, (1, 0, v, 0, 1, 0), RESAMPLE_MODE)


def TranslateY(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, 0, 0, 1, v),
        RESAMPLE_MODE)


def TranslateXConst(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() > 0.5:
        v = -v
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, v, 0, 1, 0),
        RESAMPLE_MODE)


def TranslateYConst(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() > 0.5:
        v = -v
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, 0, 0, 1, v),
        RESAMPLE_MODE)


def _float_parameter(v, max_v) -> float:
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v) -> int:
    return int(v * max_v / PARAMETER_MAX)


def rand_augment_pool(args) -> list:
    augs = [
        # (func, max_val, bias)
        (AutoContrast, None, None),
        (Brightness, 1.8, 0.1),
        (Color, 1.8, 0.1),
        (Contrast, 1.8, 0.1),
        (CutoutConst, 40, None),
        (Equalize, None, None),
        (Invert, None, None),
        (Posterize, 4, 0),
        (Rotate, 30, None),
        (Sharpness, 1.8, 0.1),
        (ShearX, 0.3, None),
        (ShearY, 0.3, None),
        (Solarize, 256, None),
        (TranslateXConst, 100, None),
        (TranslateYConst, 100, None),
    ]
    return augs


class RandAugment(object):
    def __init__(
            self,
            args,
            n: int = 2,
            m: int = 10,
            resample_mode=Image.BILINEAR) -> None:
        assert n >= 1
        assert m >= 1
        global RESAMPLE_MODE
        RESAMPLE_MODE = resample_mode
        self.n = n
        self.m = m
        self.augment_pool = rand_augment_pool(args)

    def __call__(self, img: Image) -> Image:
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        return img
