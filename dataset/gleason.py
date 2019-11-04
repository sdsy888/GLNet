"""Load Gleason2019 dataset"""

import os
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])


def find_label_map_name(img_filenames, labelExtension=".png"):
    img_filenames = img_filenames.replace('_sat.jpg', '_mask')
    return img_filenames + labelExtension


def RGB_mapping_to_class(label):
    l, w = label.shape[0], label.shape[1]
    classmap = np.zeros(shape=(l, w))
    indices = np.where(np.all(label == (0, 255, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 1
    indices = np.where(np.all(label == (255, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 2
    indices = np.where(np.all(label == (255, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 3
    indices = np.where(np.all(label == (0, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 4
    indices = np.where(np.all(label == (0, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 5
    indices = np.where(np.all(label == (255, 255, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 6
    indices = np.where(np.all(label == (0, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 0
    #     plt.imshow(colmap)
    #     plt.show()
    return classmap


def classToRGB(label):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3)).astype(np.float32)
    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 255]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 0]
    indices = np.where(label == 3)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 255]
    indices = np.where(label == 4)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 0]
    indices = np.where(label == 5)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 255]
    indices = np.where(label == 6)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 255]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]
    transform = ToTensor()
    #     plt.imshow(colmap)
    #     plt.show()
    return transform(colmap)


def class_to_target(inputs, numClass):
    batchSize, l, w = inputs.shape[0], inputs.shape[1], inputs.shape[2]
    target = np.zeros(shape=(batchSize, l, w, numClass), dtype=np.float32)
    for index in range(7):
        indices = np.where(inputs == index)
        temp = np.zeros(shape=7, dtype=np.float32)
        temp[index] = 1
        target[indices[0].tolist(), indices[1].tolist(), indices[2].tolist(), :] = temp
    return target.transpose(0, 3, 1, 2)


def label_bluring(inputs):
    batchSize, numClass, height, width = inputs.shape
    outputs = np.ones((batchSize, numClass, height, width), dtype=np.float)
    for batchCnt in range(batchSize):
        for index in range(numClass):
            outputs[batchCnt, index, ...] = cv2.GaussianBlur(inputs[batchCnt, index, ...].astype(np.float), (7, 7), 0)
    return outputs


class Gleason(data.Dataset):
    """input and label image dataset"""
    def __init__(self, file_index='./data/file_index.csv', label=False, transform=False):
        super(Gleason, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.label = label
        self.transform = transform
        self.file_index = file_index
        self.classdict = {
            1: "benign",
            # 2: "agriculture",
            3: "severe",
            4: "more_severe",
            5: "most_severe",
            0: "background"
        }
        self.file_list = []
        with open(self.file_index, 'r') as fp:
            self.file_list = fp.read().splitlines()
            self.file_list = [(i.split(',')[0], i.split(',')[1]) for i in self.file_list]

        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04)
        # using bilinear could lead to new label
        # self.resizer = transforms.Resize((5120, 5120), Image.BILINEAR)
        self.resizer = transforms.Resize((5120, 5120), Image.NEAREST)

    def __getitem__(self, index):
        sample = {}
        img_fn = self.file_list[index][0]
        label_fn = self.file_list[index][1]
        sample['id'] = img_fn[0][:16]  # only the file name: slide001_core005
        image = self.resizer(Image.open(img_fn))  # w, h
        sample['image'] = image
        # print('%%%%', sample['image'].size)

        # sample['image'] = transforms.functional.adjust_contrast(image, 1.4)
        if self.label:
            # label = scipy.io.loadmat(join(self.root, 'Notification/' + self.ids[index].replace('_sat.jpg', '_mask.mat')))["label"]
            # label = Image.fromarray(label)
            label = self.resizer(Image.open(label_fn))
            # print(label_fn, label.size)
            sample['label'] = label
            # print('****', sample['label'].size)
        if self.transform and self.label:
            image, label = self._transform(image, label)
            sample['image'] = image
            sample['label'] = label
        # return {'image': image.astype(np.float32), 'label': label.astype(np.int64)}
        # print('before', sample['image'].size, sample['label'].size)
        return sample

    def _transform(self, image, label):
        # if np.random.random() > 0.5:
        #     image = self.color_jitter(image)

        # if np.random.random() > 0.5:
        #     image = transforms.functional.vflip(image)
        #     label = transforms.functional.vflip(label)

        if np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        if np.random.random() > 0.5:
            degree = random.choice([90, 180, 270])
            image = transforms.functional.rotate(image, degree)
            label = transforms.functional.rotate(label, degree)

        # if np.random.random() > 0.5:
        #     degree = 60 * np.random.random() - 30
        #     image = transforms.functional.rotate(image, degree)
        #     label = transforms.functional.rotate(label, degree)

        # if np.random.random() > 0.5:
        #     ratio = np.random.random()
        #     h = int(2448 * (ratio + 2) / 3.)
        #     w = int(2448 * (ratio + 2) / 3.)
        #     i = int(np.floor(np.random.random() * (2448 - h)))
        #     j = int(np.floor(np.random.random() * (2448 - w)))
        #     image = self.resizer(transforms.functional.crop(image, i, j, h, w))
        #     label = self.resizer(transforms.functional.crop(label, i, j, h, w))

        return image, label

    def __len__(self):
        return len(self.file_list)
