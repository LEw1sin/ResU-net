# -*- coding: utf-8 -*-
# @Time : 2023/8/26 16:48
# @Author : ymy
# @Email : 3037845288@qq.com
# @File : contrast_stretch.py
# @Project : unet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def gamma_correction(image, gamma, max_value=255):
    # Applying gamma correction
    corrected_image = np.power(image / float(max_value), gamma) * max_value
    return corrected_image

np.set_printoptions(threshold=np.inf)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 读取图像
image = cv2.imread(r"F:\PycharmProjects\unet2\utils\data\test3\1\img0018-48.0586.png")

gamma_value = 4.5  # 伽马值，控制非线性增加的程度
# 应用非线性亮度调整
img = gamma_correction(image, gamma_value)

target_height, target_width = 232, 256
img_height, img_width = img.shape
if img_height == target_height and img_width == target_width:
    pass  # 图像尺寸已满足要求
elif img_height == target_width and img_width == target_height:
    img = cv2.transpose(img)  # 转置
elif img_height < target_height or img_width < target_width:
    # 零填充
    img = cv2.copyMakeBorder(img, 0, target_height - img_height, 0, target_width - img_width, cv2.BORDER_CONSTANT,
                             value=0)
else:
    # 裁剪
    img = img[:target_height, :target_width]
# # 限制亮的部分增加亮度
# img[img > max_brightness_increase] = max_brightness_increase
# 显示原图和调整后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Brightened Image', img)

# 等待按键响应，并关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 调整亮度
# brightness_factor = 1.5  # 调整的亮度因子，大于1增加亮度，小于1降低亮度
#
# # 应用亮度调整
# brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

# # 显示原图和调整后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Brightened Image', brightened_image)
#
# # 等待按键响应，并关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()

