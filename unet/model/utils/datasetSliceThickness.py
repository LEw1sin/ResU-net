import os
import glob
import pydicom
from torch.utils.data import Dataset


class SliceThickness_Loader(Dataset):
    def __init__(self, data_path):  # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = [[] for _ in range(len(os.listdir(data_path)))]
        # 遍历data_path下的所有子文件夹
        self.subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]
        for i, folder in enumerate(self.subfolders):
            image_paths = glob.glob(os.path.join(folder, '*.dcm'))
            for image_path in image_paths:
                self.imgs_path[i].append(image_path)  # 将每个子文件夹的图片路径列表添加到外层列表


    def __getitem__(self, indices):
        folder_index, img_index = indices
        image_path = self.imgs_path[folder_index][img_index]
        dicom_data = pydicom.dcmread(image_path)
        SliceThickness = dicom_data.SliceThickness
        return SliceThickness

    def total_len(self):
        total_length = 0
        for row in range(len(self.subfolders)):
            total_length += len(self.imgs_path[row])
        return total_length

    def folder_len(self):
        return len(self.subfolders)

# 在主函数中使用
if __name__ == "__main__":
    isbi_dataset = SliceThickness_Loader(r"unet/wly/normal control child/zhuang_xi_tong_014Y")
    print("数据集中的子文件夹数量：", isbi_dataset.folder_len())
    print("数据集中的总图片数量：", isbi_dataset.total_len())
    for i in range(len(isbi_dataset.subfolders)):
        print(f"子文件夹 {isbi_dataset.subfolders[i]} 中的图片数量：", len(isbi_dataset.imgs_path[i]))

    folder_index = 3  # 假设想访问第一个子文件夹的数据
    for img_index in range(len(isbi_dataset.imgs_path[folder_index])):
        SliceThickness = isbi_dataset[folder_index, img_index]
        print("获取到的图像断层数：", type(SliceThickness))
        

