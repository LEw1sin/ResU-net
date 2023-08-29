from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from torchvision import transforms

np.set_printoptions(threshold=np.inf)


def gamma_correction(image, gamma, max_value=255):
    image = torch.from_numpy(image).to(device=device, dtype=torch.float32)  # 将 Numpy 数组转换为张量并移动到设备上
    # 应用 gamma 校正
    corrected_image = torch.pow(image / float(max_value), gamma) * max_value
    return corrected_image





def plot_loss(train_loss, valid_loss, num_epochs):
    train_loss_np = [loss.detach().numpy() for loss in train_loss]
    valid_loss_np = [loss.detach().numpy() for loss in valid_loss]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), valid_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def train_net(net, device, data_path, valid_path, epochs=40, batch_size=1, lr=0.00001):

    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path, )
    valid_dataset = ISBI_Loader(valid_path, )
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    train_total_loss = []
    valid_total_loss = []
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        train_losses = 0
        valid_losses = 0
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # 进行亮度调整
            gamma_value = 4.5
            image = gamma_correction(image.cpu().numpy(), gamma_value)  # 先转换回 Numpy 数组，然后进行 gamma 校正

            target_height, target_width = 232, 256
            img_height, img_width = image.shape[2], image.shape[3]  # Assuming image is a 4D tensor (batch_size, channels, height, width)
            if img_height != target_height or img_width != target_width:
                # 进行裁剪或零填充操作
                if img_height == target_width and img_width == target_height:
                    image = image.transpose(2, 3)  # 转置
                    label = label.transpose(2, 3)  # 转置
                elif img_height < target_height or img_width < target_width:
                    pad_top = max((target_height - img_height) // 2, 0)
                    pad_bottom = target_height - img_height - pad_top
                    pad_left = max((target_width - img_width) // 2, 0)
                    pad_right = target_width - img_width - pad_left
                    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
                    label = F.pad(label, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
                else:
                    image = image[:, :, :target_height, :target_width]  # 裁剪
                    label = label[:, :, :target_height, :target_width]  # 裁剪


            # 使用网络参数，输出预测结果
            pred = net(image)
            print(pred)
            # 计算loss
            loss = criterion(pred, label)
            train_losses += loss
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model2.pth')
            # 更新参数
            loss.backward()
            optimizer.step()

        # for image,label in valid_loader:
        #     optimizer.zero_grad()
        #     # 将数据拷贝到device中
        #     image = image.to(device=device, dtype=torch.float32)
        #     label = label.to(device=device, dtype=torch.float32)
        #     # 使用网络参数，输出预测结果
        #     pred = net(image)
        #     # print(pred)
        #     # 计算loss
        #     loss = criterion(pred, label)
        #     valid_losses+=loss
        #     print('validation loss',loss.item())
        # train_losses/=len(train_loader)
        # train_total_loss.append(train_losses)
        # valid_losses/=len(valid_loader)
        # valid_total_loss.append(valid_losses)
        # plot_loss(train_loss=train_total_loss, valid_loss=valid_total_loss, num_epochs=epoch+1)


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)

    # 指定训练集地址，开始训练
    data_path = "utils/data/train4/"
    valid_data_path = "utils/data/valid4/"
    train_net(net, device, data_path, valid_data_path, epochs=40, batch_size=1)
