import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from tools import proportion,label_change,judge
from tqdm import tqdm
category = [[0,1,2,6,7,8,9],[3,4,5]]
# category = [[0,1,2,3],[4,5,6,7,8,9]]
# category = [[0,2,4,6],[1,3,5,7,8,9]]
# category = [[3,4,5,7],[0,1,2,6,8,9]]
category = [[0,1,3,4,5,8,9],[2,6,7]]
category = [[0,1,3,8,9],[2,4,5,6,7]]
category = [[0,1,2,4,5,6,7,8,9],[3]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='datasets', train=True,
                                         download=True, transform=transform)
# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set, batch_size=50,
                                           shuffle=True, num_workers=0)

# 10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                        download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=False, num_workers=0)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()
#




net = LeNet()  # 定义训练的网络模型
net.to(device)

a = net.parameters()
loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
# 定义优化器（训练参数，学习率）
optimizer = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(15):  # 一个epoch即对整个训练集进行一次训练
    running_loss = 0.0
    time_start = time.perf_counter()

    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
        inputs, labels = data  # 获取训练集的图像和标签
        optimizer.zero_grad()  # 清除历史梯度

        tag1 = []
        tag2 = []
        tag1img = []
        tag2img = []
        tag1num = 0
        tag2num = 0
        for i in range(len(labels)):
            if (judge(labels[i],category[0])):
                tag1img.append(i)
                # tag1num += 1
                # if (labels[i] == 0 or labels[i] == 1):
                #     tag1.append(int(labels[i]))
                # else:
                #     tag1.append(int(labels[i]) - 6)
            else:
                tag2.append(int(labels[i]))
                tag2img.append(i)
                tag2num += 1
        images_cls1 = torch.zeros(tag1num, 3, 32, 32)
        images_cls2 = torch.zeros(tag2num, 3, 32, 32)
        label_cls1 = torch.zeros(tag1num).long()
        label_cls2 = torch.zeros(tag2num).long()
        for i in range(tag2num):
            images_cls2[i] = inputs[tag2img[i]]
            label_cls2[i] = tag2[i]


        outputs = net(images_cls2.to(device))  # 正向传播
        loss = loss_function(outputs, label_cls2.to(device))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数

        # 打印耗时、损失、准确率等数据
        running_loss += loss.item()
        predict_yno = []
        acc = 0.0
        if step % 1000 == 999:  # print every 1000 mini-batches，每1000步打印一次
            with torch.no_grad():
                val_bar = tqdm(test_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    if (judge(val_labels[0],category[1])):
                        outputs = net(val_images.to(device))
                        predict_y = torch.max(outputs, dim=1)[1]
                        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                        if(predict_y[0] != val_labels[0]):
                            predict_yno.append(int(predict_y[0]))


            val_accurate = acc / (len(category[1]) * 1000)
            pre_no = proportion(predict_yno)
            print(pre_no)
            print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                  (epoch + 1, step + 1, running_loss / 500, val_accurate))

            print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
            running_loss = 0.0

print('Finished Training')

# 保存训练得到的参数
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)