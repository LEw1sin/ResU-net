from model.unet_model import UNet
from model.utils.dataset import Pretrain_Loader
from model.utils.dice_scores import dice_loss
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from torchvision import transforms
from model.image_trans import ImageTransform
from model.label_trans import LabelTransform
import os

np.set_printoptions(threshold=np.inf)


def plot(variable_1,variable_1_label, variable_2,variable_2_label, num_epochs, file_path,str_x,str_y,str_title):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), variable_1, label=variable_1_label)
    plt.plot(range(1, num_epochs + 1), variable_2, label=variable_2_label)
    plt.xlabel(str_x)
    plt.ylabel(str_y)
    plt.title(str_title)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.show()

def train_net(net, device, data_path, valid_path, epochs=80, batch_size=32, lr=0.0001):

    image_transform = transforms.Compose([
        ImageTransform(target_height=180, target_width=224, gamma=0.5),
        transforms.ToTensor()
    ])

    label_transform = transforms.Compose([
        LabelTransform(target_height=180, target_width=224),
        transforms.ToTensor()
    ])

    # 加载训练集和验证集
    pretrain_dataset = Pretrain_Loader(data_path, image_transform=image_transform, label_transform=label_transform)
    valid_dataset = Pretrain_Loader(valid_path, image_transform=image_transform, label_transform=label_transform)
    train_loader = torch.utils.data.DataLoader(dataset=pretrain_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr,
                              weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    train_total_loss = []
    valid_total_loss = []
    train_MPA = [] # 平均像素精度MPA
    train_mIOU = []
    train_F1_Score = []
    valid_MPA = [] # 平均像素精度MPA
    valid_mIOU = []
    valid_F1_Score = []
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        train_losses = 0  # 训练损失
        p00_total=0
        p01_total=0
        p10_total=0
        p11_total=0
        n = 0  # 训练样本总数
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)



            # 使用网络参数，输出预测结果
            pred = net(image)
            # print(pred)
            # 计算loss
            loss = criterion(pred.squeeze(1), label.squeeze(1).float())
            loss += dice_loss(F.sigmoid(pred.squeeze(1)), label.squeeze(1).float(), multiclass=False)
            train_losses += loss
            # 评估模型分割性能
            train_possibility = F.sigmoid(pred.squeeze(1))
            p00 = ((train_possibility <= 0.7) & (label.squeeze(1) == 0)).sum() # TN
            p01 = ((train_possibility > 0.7) & (label.squeeze(1) == 0)).sum() # FN
            p11 = ((train_possibility > 0.7) & (label.squeeze(1) == 1)).sum() # TP
            p10 = ((train_possibility <= 0.7) & (label.squeeze(1) == 1)).sum() # FP

            p00_total += p00
            p01_total += p01
            p11_total += p11
            p10_total += p10
            
            # 更新参数
            loss.backward()
            optimizer.step()
            n += 1
        # 训练完成后，求本epoch的训练参数
        # 平均像素精度MPA
        MPA = 0.5 * (p00_total/(p01_total+p00_total)+p11_total/(p10_total+p11_total))
        # 平均交并比mIOU
        mIOU = p11_total/(p10_total+p11_total+p01_total)
        # F1-score
        precision = p11_total/(p10_total+p11_total)
        recall = p11_total/(p01_total+p11_total)
        F1_Score = 2*((precision*recall)/(precision+recall))
        train_losses /= len(train_loader)
        train_total_loss.append(train_losses.item())  # 将平均训练损失的值添加到列表中
        train_MPA.append(MPA.item())
        train_mIOU.append(mIOU.item())
        train_F1_Score.append(F1_Score.item())
        print('train_loss:' + str(train_losses.item()) + f' in epoch{epoch}')
        print('train_MPA:' + str(MPA.item()) + f' in epoch{epoch}')
        print('train_mIOU:' + str(mIOU.item()) + f' in epoch{epoch}')
        print('train_F1_Score:' + str(F1_Score.item()) + f' in epoch{epoch}')


        # 验证
        net.eval()
        valid_losses = 0
        p00_total=0
        p01_total=0
        p10_total=0
        p11_total=0
    
        with torch.no_grad():
            for image, label in valid_loader:
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)



                # 使用网络参数，输出预测结果
                pred = net(image)
                # 计算loss
                loss = criterion(F.sigmoid(pred.squeeze(1)), label.squeeze(1).float())
                loss += dice_loss(F.sigmoid(pred.squeeze(1)), label.squeeze(1).float(), multiclass=False)
                valid_losses += loss
                
                valid_possibility = F.sigmoid(pred.squeeze(1))
                p00 = ((valid_possibility <= 0.7) & (label.squeeze(1) == 0)).sum()
                p01 = ((valid_possibility > 0.7) & (label.squeeze(1) == 0)).sum()
                p11 = ((valid_possibility > 0.7) & (label.squeeze(1) == 1)).sum()
                p10 = ((valid_possibility <= 0.7) & (label.squeeze(1) == 1)).sum()

                p00_total += p00
                p01_total += p01
                p11_total += p11
                p10_total += p10


        MPA = 0.5 * (p00_total/(p01_total+p00_total)+p11_total/(p10_total+p11_total))
        # 平均交并比mIOU
        mIOU = p11_total/(p10_total+p11_total+p01_total)
        # F1-score
        precision = p11_total/(p10_total+p11_total)
        recall = p11_total/(p01_total+p11_total)
        F1_Score = 2*((precision*recall)/(precision+recall))

        valid_losses /= len(valid_loader)
        valid_total_loss.append(valid_losses.item())
        valid_MPA.append(MPA.item())
        valid_mIOU.append(mIOU.item())
        valid_F1_Score.append(F1_Score.item())

        print('valid_loss:' + str(valid_losses.item()) + f' in epoch{epoch}')
        print('valid_MPA:' + str(MPA.item()) + f' in epoch{epoch}')
        print('valid_mIOU:' + str(mIOU.item()) + f' in epoch{epoch}')
        print('valid_F1_Score:' + str(F1_Score.item()) + f' in epoch{epoch}')

        if best_loss > valid_losses:
            best_loss = valid_losses
            print(f'save the model in epoch{epoch}')
            torch.save(net.state_dict(), 'test_model_best.pth')
        elif epoch == epochs-1:
            torch.save(net.state_dict(), 'test_model_last.pth')


    plot_loss_file_path = os.path.join(base_path, "loss_curve_0.5_old.png")
    plot_MPA_file_path = os.path.join(base_path, "MPA_curve_0.5_old.png")
    plot_mIOU_file_path = os.path.join(base_path, "mIOU_curve_0.5_old.png")
    plot_F1_Score_file_path = os.path.join(base_path, "F1_Score_curve_0.5_old.png")
    plot(train_total_loss, 'train_loss',valid_total_loss,'valid_loss',epochs, plot_loss_file_path,'epochs','Loss','Training and Validation Loss over Iterations')
    plot(train_MPA, 'train_MPA',valid_MPA,'valid_MPA',epochs, plot_MPA_file_path,'epochs','MPA','Training and Validation MPA over Iterations')
    plot(train_mIOU, 'train_mIOU',valid_mIOU,'valid_mIOU',epochs, plot_mIOU_file_path,'epochs','mIOU','Training and Validation mIOU over Iterations')
    plot(train_F1_Score, 'train_F1_Score',valid_F1_Score,'valid_F1_Score',epochs, plot_F1_Score_file_path,'epochs','F1_Score','Training and Validation F1_Score over Iterations')


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # # 加载网络，图片单通道1，分类为1。
    # net = UNet(n_channels=1, n_classes=1)
    # # 将网络拷贝到deivce中
    # net.to(device=device)
    # 检查可用的CUDA设备数量
    num_cuda_devices = torch.cuda.device_count()
    print("可用的CUDA设备数量：", num_cuda_devices)

    # 创建模型并将其复制到所有可用的GPU上
    net = UNet(n_channels=1, n_classes=1)
    if num_cuda_devices > 1:
        net = nn.DataParallel(net)

    # 将模型移动到GPU上
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # 指定训练集地址，开始训练
    base_path = os.path.dirname(os.path.realpath(__file__))  # 获取当前脚本的绝对路径
    data_path = os.path.join(base_path, "model/utils/data/train5")
    valid_data_path = os.path.join(base_path, "model/utils/data/valid5")

    train_net(net, device, data_path, valid_data_path, epochs=50, batch_size=32)


