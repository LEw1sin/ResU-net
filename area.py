import os
import cv2
import numpy as np

# 文件夹路径
folder_path = r'F:\PycharmProjects\unet\data'

# 设置阈值
threshold = 255

# 遍历文件夹中的每张图像
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(folder_path, filename)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 将图像二值化，得到白色区域为True，其他区域为False
        binary_image = (gray_image == threshold)

        # 假设图像中每个像素代表实际面积（如1平方单位）
        unit_area = 1.0

        # 计算白色区域的面积（矩形法）
        white_area_rect = np.sum(binary_image) * unit_area
        print(f"图像 {filename} 的白色区域面积（矩形法）：{white_area_rect}")

        # 计算白色区域的总面积（多边形拟合法）
        contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到白色区域的轮廓
        areas = np.array([cv2.contourArea(contour) for contour in contours])
        total_area_polygon = np.sum(areas) * unit_area
        print(f"图像 {filename} 的白色区域面积（多边形拟合法）：{total_area_polygon}")

        # 创建一个彩色图像副本，用于绘制轮廓
        colored_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        # 在图像上绘制轮廓
        cv2.drawContours(colored_image, contours, -1, (0, 0, 255), 1)  # 最后两个参数分别是颜色和线宽
        # 显示绘制了轮廓的图像
        cv2.imshow(f"Contours of {filename}", colored_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
