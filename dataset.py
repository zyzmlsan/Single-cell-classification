import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import os
import time
import random
import numpy as np
import pickle
from PIL import Image
from PIL import ImageFile
from libtiff import TIFF
import cv2

input_size = 224
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop((140, 140)),
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.RandomCrop((140, 140)),
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
    ]),
}

calibration = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
totensor = transforms.ToTensor()

data_dir = 'data'
file_list = [data_dir + '/' + filename for filename in os.listdir(data_dir)]
train_list = random.sample(file_list, int(len(file_list) * 0.9))
test_list = list(set(file_list) - set(train_list))

class SRSSet(Dataset):
    def __init__(self, phase):
        super(SRSSet, self).__init__()
        self.phase = phase
        self.data_dir = data_dir
        self.file_list = self.list_file()

    def list_file(self):
        # need to devide train and test sets, they are oerlapped now
        if self.phase == 'train':
            file_list = train_list
        else:
            file_list = test_list
        return file_list

    def __getitem__(self, item):
        # 假设只有两类
        direc = self.file_list[item]
        seed = int(time.time() * 1e3)
        # random seed, to make sure the label and datum are rotated synchronied
        for file in os.listdir(direc):
            if 'pkl' in file:
                pass
                # with open(direc + '/' + file, 'rb') as f:
                #     datum = pickle.load(f)

            elif 'label' in file:
                label = Image.open(direc + '/' + file)
                torch.manual_seed(seed)
                label = data_transforms[self.phase](label)
                label = np.array(label)
                if 'Drug1' in direc:
                    label[label == 1] = 2

            elif 'img' in file:
                datum = Image.open(direc + '/' + file)
                torch.manual_seed(seed)
                datum = data_transforms[self.phase](datum)
                datum = totensor(datum)
                datum = calibration(datum)

        return datum, label

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    c = SRSSet('train')
    print(c.__len__())
    print('train', len(c.file_list), 'test', len(test_list))
    datum, label = c.__getitem__(2)
    print(datum.shape, label.shape)
    for i in range(224):
        print(label[i])