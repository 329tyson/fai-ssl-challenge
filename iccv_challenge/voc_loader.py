import os

import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from utils import getlogger
from utils import DatasetConfig
from datasets.pil_augmentation import get_image
from datasets.transforms import normalize


class VOC_DATASET(data.Dataset):
    def __init__(self, config: DatasetConfig):
        self.image_path = config.imagepath
        self.label_path = config.labelpath
        self.preprocess = transforms.Compose(config.preprocess)
        self.images = []

        self.load_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imagepath, label = self.images[index]
        image = get_image(imagepath)

        return self.preprocess(image), label

    def load_images(self):
        # for fai-ssl-challenge dataset, you need to use numpy formatted image, label file
        assert self.image_path.endswith(".npy")
        assert self.label_path.endswith(".npy")

        image_file = np.load(self.image_path)
        label_file = np.load(self.label_path)

        for image, label in zip(image_file, label_file):
            label[label == -1] = 0
            self.images.append((image, label))


def _voc_loader(image_path: str, label_path: str, batch_size=128, shuffle=True):
    '''
    Args:
        image_path : path to numpy file describes images
        label_path : path to numpy file describes labels
    '''
    logger = getlogger()
    image_file = np.load(image_path)
    label_file = np.load(label_path)

    logger.info(f"Loading image from {image_path}")
    logger.info(f"\t corresponding label from {label_path}")

    # imagefile and labelfile are should be the same in length
    assert len(image_file) == len(label_file)

    # define data augmentation
    preprocess = [
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize(),
    ]

    logger.info(f"Applying {preprocess}")

    # build dataset
    voc_dataset = VOC_DATASET(
        DatasetConfig(
            imagepath=image_path,
            labelpath=label_path,
            preprocess=preprocess,
        )
    )

    # build dataloader and return
    return data.DataLoader(
        voc_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=8,
    )


def load_voc(type: str):
    voc_train_image_path = os.environ.get("VOC_TRAIN_IMAGE")
    voc_train_label_path = os.environ.get("VOC_TRAIN_LABEL")

    voc_val_image_path = os.environ.get("VOC_VAL_IMAGE")
    voc_val_label_path = os.environ.get("VOC_VAL_LABEL")

    voc_test_image_path = os.environ.get("VOC_TEST_IMAGE")
    voc_test_label_path = os.environ.get("VOC_TEST_LABEL")

    type = type.lower()
    if type == "train":
        return _voc_loader(voc_train_image_path, voc_train_label_path)

    elif type == "val":
        return _voc_loader(voc_val_image_path, voc_val_label_path)

    elif type == "test":
        return _voc_loader(voc_test_image_path, voc_test_label_path)

    else:
        raise NotImplementedError
