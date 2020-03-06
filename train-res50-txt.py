# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

from RandomErasing import RandomErasing


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
            print("classe number: {}".format(len(classes)))
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], ),
    RandomErasing(probability=0.5, mean=[0, 0, 0])
])
val_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = LoadDataset(txtdata="/data/train.txt",
                              dir_root='/data/',
                              transform=train_transforms)
val_dataset = LoadDataset(txtdata="/data/val.txt",
                            dir_root='/data/',
                            transform=val_transforms)

train_dataloaders = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_dataloaders = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

train_dataset_sizes = len(train_dataset)
val_dataset_sizes = len(val_dataset)
class_names = train_dataset.classes
print(train_dataset.class_to_idx)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, labels, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # compute output
            output = model(inputs)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            if i % 500 == 0:
                print('Test: [{0}/{1}], '
                      'Loss(avg): {loss.val:.4f}({loss.avg:.4f}), '
                      'Top1 acc(avg): {top1.val:.3f}({top1.avg:.3f}), '
                      'Top5 acc(avg): {top5.val:.3f}({top5.avg:.3f})'.format(
                    i, len(val_loader), loss=losses,
                    top1=top1, top5=top5))
        print(' * Top1 avg_acc {top1.avg:.3f} , Top5 avg_acc {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def train(train_loader, model, criterion, optimizer, epoch, num_epochs):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (inputs, labels) in enumerate(train_loader):
        # measure data loading time

        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute output
        output = model(inputs)
        loss = criterion(output, labels)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print('Epoch: [{0}/{1}][{2}/{3}], '
                  'Loss(avg): {loss.val:.4f}({loss.avg:.4f}), '
                  'Top1 acc(avg): {top1.val:.3f}({top1.avg:.3f}), '
                  'Top5 acc(avg): {top5.val:.3f}({top5.avg:.3f})'.format(
                epoch, num_epochs, i, len(train_loader),
                loss=losses, top1=top1, top5=top5))


def main():
    if not os.path.exists('weights'):
        os.makedirs('weights')
    model = models.resnet18(pretrained=True)
    freeze_conv_layer = False
    if freeze_conv_layer:
        for param in model.parameters():  # freeze layers
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)
    # print(model)
    # from torchsummary import summary
    # summary(model, (3, 224, 224))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if freeze_conv_layer:
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 25
    for epoch in range(num_epochs):
        scheduler.step()
        train(train_dataloaders, model, criterion, optimizer, epoch, num_epochs)
        acc = validate(val_dataloaders, model, criterion)
        torch.save(model.state_dict(), ('weights/Epoch{}_acc{:.2f}.pt'.format(epoch, acc)))

    return model


def predict():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)
    model.load_state_dict(torch.load('weights/Epoch4_acc98.91.pt'))
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print("batch %d" % i)
            for j in range(inputs.size()[0]):
                print(
                    "{} pred label:{}, true label:{}".format(len(preds), class_names[preds[j]], class_names[labels[j]]))


if __name__ == "__main__":
    main()
    # predict()
