import os
import cv2
import glob
import pydicom
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from unet.model.image_trans import ImageTransform
import matplotlib.pyplot as plt

def visualize_images(image):
    # Convert tensor to numpy array
    image = image.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Processed Image')

    plt.show()

class Predict_Loader(Dataset):
    def __init__(self, data_path, image_transform=None):  # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = [[] for _ in range(len(os.listdir(data_path)))]
        self.transform = image_transform
        # 遍历data_path下的所有子文件�?
        self.subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]
        for i, folder in enumerate(self.subfolders):
            image_paths = glob.glob(os.path.join(folder, '*.dcm'))
            for image_path in image_paths:
                self.imgs_path[i].append(image_path)  # 将每个子文件夹的图片路径列表添加到外层列�?
            self.imgs_path[i] = sorted(self.imgs_path[i])
        

    def __getitem__(self, indices):
        folder_index, img_index = indices
        image_path = self.imgs_path[folder_index][img_index]
        dicom_data = pydicom.dcmread(image_path)
        image = dicom_data.pixel_array

        if self.transform is not None:
            image= self.transform(image)
        return image

    def total_len(self):
        total_length = 0
        for row in range(len(self.subfolders)):
            total_length += len(self.imgs_path[row])
        return total_length

    def folder_len(self):
        return len(self.subfolders)

# 在主函数中使�?
if __name__ == "__main__":

    transform = transforms.Compose([
        ImageTransform(target_height=180, target_width=224, gamma=0.6),
        transforms.ToTensor(),
    ])
    predict_dataset = Predict_Loader("unet/wly/normal control child/zhuang_xi_tong_014Y", image_transform=transform)
    print("数据集中的子文件夹数量：", predict_dataset.folder_len())
    print("数据集中的总图片数量：", predict_dataset.total_len())
    for i in range(len(predict_dataset.subfolders)):
        print(f"子文件夹 {predict_dataset.subfolders[i]} 中的图片数量�?", len(predict_dataset.imgs_path[i]))

        if i == 0:
            print(predict_dataset.imgs_path[i])
            pass
    print(predict_dataset.imgs_path[2][0])
    folder_index = 3  # 假设想访问第一个子文件夹的数据
    for img_index in range(len(predict_dataset.imgs_path[folder_index])):
        image = predict_dataset[folder_index, img_index]
        print("获取到的图像维度�?", image.shape)
        if img_index == 5:
            print(image)
            average_gray_value = image.mean()
            print("图像的平均灰度值：", average_gray_value)
            visualize_images(image)
            print(predict_dataset.subfolders[folder_index])
