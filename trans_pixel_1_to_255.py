# -*- coding: utf-8 -*-
# @Time : 2023/8/17 21:42
# @Author : ymy
# @Email : 3037845288@qq.com
# @File : trans_pixel_1_to_255.py
# @Project : unet

from PIL import Image
import os
import numpy as np

np.set_printoptions(threshold=np.inf)

dir_path="C:\\Users\\30378\\Desktop\\unet\\utils\\data\\train4\\label"
output_path="C:\\Users\\30378\\Desktop\\转换后的标签"
for root, dirs, files in os.walk(dir_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        if filename.endswith(".png"):
            # 读取图像
            img = Image.open(file_path)
            # 获取图像像素值（二维数组）
            pixels = np.array(img)
            #trans_pixiel from 1 to 255
            print(pixels)
            # pixels[pixels==1]=255
            # pil_image = Image.fromarray(pixels.astype('uint8'))
            # out_path=os.path.join(output_path, filename)
            # print(out_path)
            # pil_image.save(out_path)
print("转换完毕")