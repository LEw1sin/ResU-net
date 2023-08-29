import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet

np.set_printoptions(threshold=np.inf)


def gamma_correction(image, gamma, max_value=255):
    # Applying gamma correction
    corrected_image = np.power(image / float(max_value), gamma) * max_value
    return corrected_image


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model2.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    # tests_path = glob.glob('utils/data/test2/*.png')
    tests_path = glob.glob(r"F:\PycharmProjects\unet2\utils\data\test3\1\*.png")

    # 遍历所有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


        gamma_value = 4.5  # 伽马值，控制非线性增加的程度
        # 应用非线性亮度调整
        img = gamma_correction(img, gamma_value)


        target_height, target_width = 232, 256
        img_height, img_width = img.shape
        if img_height == target_height and img_width == target_width:
            pass  # 图像尺寸已满足要求
        elif img_height == target_width and img_width == target_height:
            img = cv2.transpose(img)  # 转置
        elif img_height < target_height or img_width < target_width:
            # 零填充
            img = cv2.copyMakeBorder(img, 0, target_height - img_height, 0, target_width - img_width, cv2.BORDER_CONSTANT, value=0)
        else:
            # 裁剪
            img = img[:target_height, :target_width]


        # 转为batch为1，通道为1的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # # 调整亮度
        # brightness_factor = 1.5 # 调整的亮度因子，大于1增加亮度，小于1降低亮度

        # # 应用亮度调整
        # img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        # 转为tensor
        # img_tensor = torch.from_numpy(img)
        # # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        # img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # # 预测
        # pred = net(img_tensor)
        # # pred = torch.sigmoid(pred)
        # # 提取结果
        # pred = np.array(pred.data.cpu()[0])[0]
        # # 处理结果
        # # print("pred",pred)
        # # print("pred.mean",pred.mean())
        # pred[pred >= -5] = 255
        # pred[pred < -5] = 0
        # # print(np.sum(pred==255))
        # # print("pred",pred)
        # # 保存图片
        # cv2.imwrite(save_res_path, pred)
        # print(save_res_path)
        cv2.imwrite(save_res_path, img)
        print(img.shape)