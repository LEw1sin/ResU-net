import numpy as np
import cv2

class ImageTransform:
    def __init__(self, target_height, target_width, gamma):
        self.target_height = target_height
        self.target_width = target_width
        self.gamma = gamma
        # 返回的是 numpy数组

    def __call__(self, img):
        img = self.size_correction(img)
        img = self.gamma_correction(img)
        return img


    # 尺寸修正
    def size_correction(self, img):
        img_height, img_width = img.shape
        if img_height > img_width:
            img = np.transpose(img, (1, 0))  # 转置维度，调整为(宽度, 高度)

        if img_height < self.target_height:
            pad_height = self.target_height - img_height
            img = np.pad(img, ((0, pad_height), (0, 0)), mode='constant')
        if img_width < self.target_width:
            pad_width = self.target_width - img_width
            img = np.pad(img, ((0, 0), (0, pad_width)), mode='constant')

        img = img[:self.target_height, :self.target_width]
        return img

    # 灰度分布修正，仿射变换和gamma变换
    def gamma_correction(self, img):
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        epsilon = 1e-6  # 一个小的正数，用于避免除以零
        img = img + epsilon  # 将图像中的所有值加上 epsilon，确保不包含零值
        corrected_img = ((img / img.max()) ** self.gamma) * 255.0

        # 剂量补偿
        if corrected_img.mean() < 30:
            non_zero_pixels = corrected_img != corrected_img.min()  # 创建一个布尔数组，表示灰度值不为最小值的像素
            corrected_img[non_zero_pixels] = np.clip(corrected_img[non_zero_pixels] + 50, 0, 255)

        return corrected_img /255.0


# if __name__ == '__main__':
    # # 示例使用
    # transform = CustomTransform(target_height=180, target_width=224, gamma=1.5)
    # transformed_image = transform(image)
