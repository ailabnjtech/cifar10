import os
import json
import torchvision
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34
from tools import proportion, label_change,judge
category = [[0,1,2,8,9],[3,4,5,6,7]]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_set = torchvision.datasets.CIFAR10(root='../datasets', train=True,
                                             download=True, transform=data_transform["train"])
    # 加载训练集，实际过程需要分批次（batch）训练
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10,
                                               shuffle=True, num_workers=0)

    # 10000张测试图片
    test_set = torchvision.datasets.CIFAR10(root='../datasets', train=False,
                                            download=False, transform=data_transform["val"])
    val_num = len(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=False, num_workers=0)

    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 10)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 10
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        time_start = time.perf_counter()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data

            tag1 = []
            tag2 = []
            tag1img = []
            tag2img = []
            tag1num = 0
            tag2num = 0

            for i in range(len(labels)):
                if (judge(labels[i], category[0])):
                    tag1img.append(i)
                    tag1num += 1
                    tag1.append(int(labels[i]))
                else:
                    tag2.append(int(labels[i]))
                    tag2img.append(i)
                    tag2num += 1
            images_cls1 = torch.zeros(tag1num, 3, 224, 224)
            label_cls1 = torch.zeros(tag1num).long()

            for i in range(tag1num):
                images_cls1[i] = images[tag1img[i]]
                label_cls1[i] = tag1[i]

            optimizer.zero_grad()
            logits = net(images_cls1.to(device))
            loss = loss_function(logits, label_cls1.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        predict_yno = []
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                if (judge(val_labels[0], category[0])):
                    outputs = net(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    # if(predict_y[0] == 2 or predict_y[0]==3):
                    #     predict_y[0] = predict_y[0] + 6
                    predict_yno.append(int(predict_y[0]))
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / (len(category[0]) * 1000)
            pre_no = proportion(predict_yno)
            print(pre_no)
            print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                  (epoch + 1, step + 1, running_loss / 500, val_accurate))

            print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
            running_loss = 0.0



        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
