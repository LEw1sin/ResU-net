import numpy as np
import cv2

class LabelTransform:
    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width
        # 返回的是 numpy数组

    def __call__(self, img):
        img = self.size_correction(img)
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
        return img.astype(np.float32)/255.0 



# if __name__ == '__main__':
    # # 示例使用
    # transform = CustomTransform(target_height=180, target_width=224, gamma=1.5)
    # transformed_image = transform(image)
