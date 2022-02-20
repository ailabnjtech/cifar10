import copy
import torch

def proportion(a):
    count=[]
    for i in range(10):
        num = 0
        for j in range(len(a)):
            if(a[j]==i):
                num +=1
        count.append(num)
    return count

def label_change(labels,category):            #大类的label转化
    labels2 = labels.clone()
    for i in range(len(labels2)):
        mark = 0
        for j in range(len(category)):
            for k in range(len(category[j])):
                if (category[j][k] == labels[i]):
                    labels2[i] = j
                    mark = 1
                    break
            if (mark == 1):
                break
    return labels2

def fine_label_change(labels,category):
    fine_label = labels.clone()
    for i in range(len(labels)):
        mark = 0
        for j in range(len(category)):
            for k in range(len(category[j])):
                if(labels[i] == category[j][k]):
                    fine_label[i] = k
                    mark = 1
                break
            if(mark == 1):
                break
    return fine_label

def judge(x,list):
    for i in range(len(list)):
        if(x == list[i]):
            return 1
    return 0

# category = [[0,1,8,9],[2,3,4,5,6,7]]
#
# images_cls1 = torch.zeros(1)
# B = judge(images_cls1,category[1])
# print(B)
# a =  judge(3,category[1])
# print(a)



