import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from PIL import Image
from RandomErasing import RandomErasing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0,1,2,3,4,5,6,7]
# 数据增强
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, mean=[0, 0, 0])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, txtdata,dir_root, transform=None):
        with open(txtdata, 'r') as f:
            imgs, all_label = [], []
            for line in f.readlines():
                line = line.strip().split(" ")
                imgs.append((line[0], line[1]))
                if line[1] not in all_label:
                    all_label.append(line[1])
            classes = set(all_label)
            #print("classe number: {}".format(len(classes)))
            classes = sorted(list(classes))
            class_to_idx = {classes[i]: i for i in range(len(classes))}  # convert label to index(from 0 to num_class-1)
            del all_label

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = imgs
        self.transform = transform
        self.root=dir_root

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        label = self.class_to_idx[label]
        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# 加载数据
batch_size = 32
num_classes = 10

data = {
    'train': LoadDataset(txtdata="/data/train.txt",
                              dir_root='/data/',
                              transform=image_transforms['train']),
    'valid': LoadDataset(txtdata="/data/val.txt",
                            dir_root='/data/',
                            transform=image_transforms['valid'])
}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])

train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)

print(train_data_size, valid_data_size)

if not os.path.exists('weights'):
    os.makedirs('weights')
# 迁移学习
resnet50 = models.resnet50(pretrained=True)
freeze_conv_layer = False
if freeze_conv_layer:
    for param in resnet50.parameters():  # freeze layers
        param.requires_grad = False
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, num_classes)

#使用multi gpus 训练
resnet50 = torch.nn.DataParallel(resnet50, device_ids=device_ids) # 声明所有可用设备
model = resnet50.to(device)

# 定义损失函数和优化器
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
if freeze_conv_layer:
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#训练 && val
def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        scheduler.step()
        for data in tqdm(train_data):
            inputs, labels = data
            # 注意数据也是放在主设备
            inputs, labels = inputs.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
            outputs = resnet50(inputs)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            train_acc += torch.sum(pred == labels.data)

        resnet50.eval()
        with torch.no_grad():
            for data in valid_data:
                inputs, labels = data
                inputs, labels = inputs.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
                outputs = resnet50(inputs)
                loss = loss_func(outputs, labels)
                _, pred = torch.max(outputs.data, 1)
                valid_loss += loss.item()
                valid_acc += torch.sum(pred == labels.data)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc.to(torch.float32) / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc.to(torch.float32) / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        torch.save(model, 'models/' + '_model_' + str(epoch + 1) + '.pt')
    return model, history


# 执行体
num_epochs = 30
trained_model, history = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
torch.save(history, 'models/'+ '_history.pt')

history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig('_loss_curve.png')
plt.show()

plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('_accuracy_curve.png')
plt.show()

