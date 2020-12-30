import os
import torchvision.transforms as transforms
from PIL import Image


def make_dataset(datapath):
    """
    Возвращаем все пути по заданной директории.
    """
    images = []
    for root, _, fnames in sorted(os.walk(datapath)):
        for name in fnames:
            path = os.path.join(root, name)
            images.append(path)
    return images


def get_transforms(args, flip=False, method=Image.BICUBIC):
    """
    Вернем Compose, подготавливающий входной тензор.
    """
    transform_list = []

    # Добавим Resize 300 -> 286
    osize = [args.load_size, args.load_size]
    transform_list.append(transforms.Resize(osize, method))

    # Добавим RandomCrop до 256
    transform_list.append(transforms.RandomCrop(args.crop_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)
