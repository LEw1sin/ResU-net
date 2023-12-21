import glob
import numpy as np
import torch
import os
import cv2
from unet.model.unet_model import UNet
import matplotlib.pyplot as plt
from unet.model.utils.datasetPredict import Predict_Loader
from torchvision import transforms
from unet.model.image_trans import ImageTransform
from unet.model.utils.datasetSliceThickness import SliceThickness_Loader
from unet.model.utils.datasetPixelSpacing import PixelSpacing_Loader
import torch.nn.functional as F


import pydicom

np.set_printoptions(threshold=np.inf)
def get_top_three_dirs(file_name):
    # 使用os.path.split()来分割路径并获取文件名
    file_dir1, file_name = os.path.split(file_name)
    # 使用os.path.split()再次分割路径以获取倒数第三级目录
    file_dir2, second_level_dir = os.path.split(file_dir1)
    _, third_level_dir = os.path.split(file_dir2)
    # 使用os.path.splitext()来分割文件名和后缀，并只保留文件名部分
    file_name_without_extension = os.path.splitext(file_name)[0]
    # 合并
    result_path = os.path.join(third_level_dir, second_level_dir, file_name_without_extension)
    return result_path

def get_PatientName(file_name):
    # 使用os.path.split()来分割路径并获取文件名
    _, PatientName = os.path.split(file_name)
    return PatientName

def get_gender(dcm_file_path):
    return pydicom.dcmread(dcm_file_path).PatientSex

def get_age(dcm_file_path):
    return pydicom.dcmread(dcm_file_path).PatientAge

def get_name(dcm_file_path):
    return pydicom.dcmread(dcm_file_path).PatientName


def visualize_images(image):
    # Convert tensor to numpy array
    # image = image.squeeze().cpu().numpy()
    image = image.squeeze().cpu().detach().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Processed Image')

    plt.show()

def recursion_mkdir(save_dir):
    if os.path.exists(save_dir) != True:
        up_dir,_ = os.path.split(save_dir)
        if up_dir == "":
            os.mkdir(save_dir)
        else:
            recursion_mkdir(up_dir)
            os.mkdir(save_dir)

def calculate_stroke_volume(volume):
    return volume.max()-volume.min()

def calculate_ejection(stroke_volume, volume):
    return stroke_volume/volume.max()

def predict(input_path, output_path, result_path):
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('test_model_best.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    image_transform = transforms.Compose([
        ImageTransform(target_height=180, target_width=224, gamma=1.0),
        transforms.ToTensor()
    ])

    predict_dataset = Predict_Loader(input_path, image_transform=image_transform)


    for row in range(len(predict_dataset.subfolders)):
        for col in range(len(predict_dataset.imgs_path[row])):
            image = predict_dataset[row, col]
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            img_tensor = image.to(device=device, dtype=torch.float32)  # 将 Numpy 数组转换为张量并移动到 device 上

            # 预测
            pred = net(img_tensor)
            mask = F.sigmoid(pred) > 0.5
            pred = mask.float() * 1
            pred = pred * 255
            # pred = pred - pred.min()
            # pred[pred != 0] = 255

            # pred[pred >= 0.7] = 255
            # pred[pred < 0.7] = 0
            # pred = ((pred - pred.min()) / (pred.max() - pred.min()) * 255)



            pred = np.array(pred.data.cpu()[0])[0]
            # 获取文件名（不包括路径）
            file_name = predict_dataset.imgs_path[row][col]
            # 将 'pred' 转换为uint8数据类型（0-255范围）
            pred_uint8 = pred.astype(np.uint8)

            save_path = os.path.join(output_path, get_top_three_dirs(file_name))
            save_dir,_ = os.path.split(save_path)
            save_name = save_path + '.png'
            recursion_mkdir(save_dir)
            # print(save_name)
            # 使用相同的文件名保存 'pred' 作为PNG图像，保存到对应的文件夹中
            cv2.imwrite(save_name, pred)

    # 体积计算部分

    threshold = 255
    patient_path = os.path.join(output_path, get_PatientName(input_path))
    folders_path = [os.path.join(patient_path, folder) for folder in os.listdir(patient_path)]
    output_dataset = [sorted(glob.glob(os.path.join(folder_path, '*.png'))) for folder_path in folders_path]

    SliceThickness_dataset = SliceThickness_Loader(input_path)
    PixelSpacing_dataset = PixelSpacing_Loader(input_path)
    # 初始化一个空的二维数组以存储每一时刻每个断层的面积，初始化一个断层厚度的二维数组，方便运算
    num_rows = int(predict_dataset.folder_len())
    num_cols = int(predict_dataset.total_len() / num_rows)
    area_array = np.zeros((num_rows, num_cols), dtype=float)
    Thickness_array = np.zeros((num_rows, num_cols), dtype=float)
    # 初始化一个空的一维数组以存储每一时刻的心腔体积
    volume_array = np.zeros((num_cols, 1), dtype=float)
    if predict_dataset.folder_len() != 8:
        print("数据集数量不是短轴数量！")
        exit(0)

    for row in range(len(output_dataset)):
        for col in range(len(output_dataset[0])):
            image_path = output_dataset[row][col]
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # 此处不能用cv读取dcm格式
            # 将图像二值化，得到白色区域为True，其他区域为False
            binary_image = (image == threshold)

            # 转换实际比例尺
            unit_area = PixelSpacing_dataset[row,col][0] * PixelSpacing_dataset[row,col][1]
            # 计算白色区域的总面积（多边形拟合法）
            _, contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 找到白色轮廓

            areas = np.array([cv2.contourArea(contour) for contour in contours])
            total_area_polygon = np.sum(areas) * unit_area
            # print(f"图像 {image_path} 的白色区域面积（多边形拟合法）：{total_area_polygon}")
            area_array[row][col] = total_area_polygon
            Thickness_array[row][col] = float(SliceThickness_dataset[row, col])

            # if row == 3 and col == 3:

            result_image = np.copy(predict_dataset[row, col].numpy().squeeze() * 255).astype(np.uint8)
            # cv2.drawContours(result_image, contours, -1, (0, 255, 0), thickness=2) # 在原图上绘制轮廓
            cv2.drawContours(result_image, contours, -1, (0, 0, 255), thickness=2) # 在原图上绘制轮廓
            file_name = predict_dataset.imgs_path[row][col]
            save_path = os.path.join(result_path, get_top_three_dirs(file_name))
            save_dir,_ = os.path.split(save_path)
            save_name = save_path + '.png'
            recursion_mkdir(save_dir)
            # print(save_name)
            # 使用相同的文件名保存 'pred' 作为PNG图像，保存到对应的文件夹中
            cv2.imwrite(save_name, result_image)

    tmp = np.sum(area_array * Thickness_array, axis=0, dtype=float)
    print(type(tmp))
    print(tmp.shape)
    print(volume_array.shape)
    volume_array += np.sum(area_array * Thickness_array, axis=0, dtype=float).reshape(25,1)
    return volume_array / 1000

if __name__ == '__main__':
    base_path = os.path.dirname(os.path.realpath(__file__))  # 获取当前脚本的绝对路径
    input_path = os.path.join(base_path,'unet/wly/normal control child/yang_zi_quan_013Y')
    output_path = os.path.join(base_path,'label')
    result_path = os.path.join(base_path,'static/result')
    # assert os.path.exists(input_path) == True
    volume = predict(input_path, output_path, result_path) * 1.7
    stroke_volume = calculate_stroke_volume(volume)
    ejection = calculate_ejection(stroke_volume, volume)
    
    


    




