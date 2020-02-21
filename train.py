import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import numpy as np
import os
import random

from dataset import SRSSet

model = smp.Unet('resnet18', classes=3, encoder_weights='imagenet')
cudaFlag = torch.cuda.is_available()

lr = 1e-5
epochs = 1000
batch_size = 10

trainset = SRSSet('train')
testset = SRSSet('test')
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


print(model)

if cudaFlag:
    model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

def get_accuracy(logit, target, batch_size):
    # print(torch.max(logit, dim=1)[1])
    corrects = (torch.max(logit, dim=1)[1].view(target.size()).data == target.long().data).sum() #虽然没有经过softmax,但是softmax是单调的
    accuracy = 100.0 * corrects / (batch_size * target.shape[1] * target.shape[2])
    return accuracy.item()

optimizer = optim.Adam([{'params': model.parameters(), 'lr': lr}])
criterion = nn.CrossEntropyLoss()
for epoch in range(epochs):
    for phase in ['train', 'test']:
        epoch_accuracy = 0
        if phase == 'train':
            model.train()
            loader = trainloader
        else:
            model.eval()
            loader = testloader
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            data, labels = batch
            if data.shape[0] < batch_size:
                continue
            if cudaFlag:
                data, labels = data.cuda(), labels.cuda()

            outputs = model(data)
            print(data.shape, outputs.shape, labels.shape)
            loss = criterion(outputs, labels.long())
            accuracy = get_accuracy(outputs, labels, batch_size)
            print('batch_', i, accuracy, loss)
            epoch_accuracy = (epoch_accuracy * i + accuracy) / (i + 1)
            if phase == 'train':
                loss.backward()
                optimizer.step()
        print(epoch_accuracy)
    # torch.save(model, 'model')
