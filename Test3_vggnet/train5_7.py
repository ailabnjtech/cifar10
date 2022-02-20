from model import vgg


import torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from tools import proportion, label_change


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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50,
                                               shuffle=True, num_workers=0)

    # 10000张测试图片
    test_set = torchvision.datasets.CIFAR10(root='../datasets', train=False,
                                            download=False, transform=data_transform["val"])
    val_num = len(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=False, num_workers=0)
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    model_name = "vgg16"

    net2 = vgg(model_name=model_name, num_classes=10, init_weights=True)
    net2.to(device)
    loss_function2 = nn.CrossEntropyLoss()
    optimizer2 = optim.Adam(net2.parameters(), lr=0.0001)

    net1 = vgg(model_name=model_name, num_classes=10, init_weights=True)
    net1.to(device)
    optimizer1 = optim.Adam(net1.parameters(), lr=0.0001)
    loss_function1 = nn.CrossEntropyLoss()

    epochs = 30
    best_acc = 0.0
    save_path1 = './{}Net1.pth'.format(model_name)
    save_path2 = './{}Net2.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net1.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data

            tag1 = []
            tag1img = []
            tag1num = 0
            tag2 = []
            tag2img = []
            tag2num = 0

            for i in range(len(labels)):
                if (labels[i] == 2 or labels[i] == 3 or labels[i] == 4 or labels[i] == 5 or labels[i] == 6 or labels[
                    i] == 7):
                    tag2img.append(i)
                    tag2num += 1
                    tag2.append(int(labels[i]) - 2)
                elif(labels[i] == 0 or labels[i] == 1 or labels[i] == 8 or labels[i] == 9):
                    tag1img.append(i)
                    tag1num += 1
                    if (labels[i] == 0 or labels[i] == 1):
                        tag1.append(int(labels[i]))
                    else:
                        tag1.append(int(labels[i]) - 6)



            images_cls1 = torch.zeros(tag1num, 3, 224, 224)
            label_cls1 = torch.zeros(tag1num).long()

            for i in range(tag1num):
                images_cls1[i] = images[tag1img[i]]
                label_cls1[i] = tag1[i]

            images_cls2 = torch.zeros(tag2num, 3, 224, 224)
            label_cls2 = torch.zeros(tag2num).long()
            for i in range(tag2num):
                images_cls2[i] = images[tag2img[i]]
                label_cls2[i] = tag2[i]

            if (tag1num != 0):
                optimizer1.zero_grad()
                logits1 = net1(images_cls1.to(device))
                loss1 = loss_function1(logits1, label_cls1.to(device))
                loss1.backward()
                optimizer1.step()
                running_loss += loss1.item()
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss1)
            if (tag2num != 0):
                optimizer2.zero_grad()
                logits2 = net2(images_cls2.to(device))
                loss2 = loss_function2(logits2, label_cls2.to(device))
                loss2.backward()
                optimizer2.step()
                running_loss += loss2.item()

        # validate
        net1.eval()
        predict_yno = []
        numx = 0
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                if (val_labels[0] == 2 or val_labels[0] == 3 or val_labels[0] == 4 or val_labels[0] == 5 or val_labels[
                    0] == 6 or val_labels[0] == 7):
                    outputs = net1(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    predict_y[0] = predict_y[0] + 2
                    numx += 1
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    # acc2 += torch.eq(predict_class2, labels2.to(device)).sum().item()
                    predict_yno.append(int(predict_y[0]))
                elif(val_labels[0] == 0 or val_labels[0] == 1 or val_labels[0] == 8 or val_labels[0] == 9):
                    outputs = net2(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    predict_y[0] = predict_y[0] + 2
                    numx += 1
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    # acc2 += torch.eq(predict_class2, labels2.to(device)).sum().item()
                    predict_yno.append(int(predict_y[0]))


                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
            pre_no = proportion(predict_yno)
            print(pre_no)
            print(numx)
            val_accurate = acc / numx
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net1.state_dict(), save_path1)
                torch.save(net2.state_dict(), save_path2)

    print('Finished Training')


if __name__ == '__main__':
    main()
