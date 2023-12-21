import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
from model.image_trans import ImageTransform
from torchvision import transforms
import numpy as np
from model.granularize_trans import GranularizeTransform

class Pretrain_Loader(Dataset):
    def __init__(self, data_path, image_transform=None, label_transform=None):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.image_transform = image_transform
        self.label_transform = label_transform  
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip


    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.label_transform is not None:
            label = self.label_transform(label)


        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


