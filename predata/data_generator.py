import pickle
import random
import matplotlib.pyplot as plt
import cv2
import math
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import torch
import os
import time
from PIL import Image
from . import datasets_processing
def seed_torch(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#在pkl文件中载入数据并进行预处理：
def load_data(items,is_train=True):
    data = []
    for item in items:
        data_item_dir = './data_input/test_pkldata/pkldata99.pkl'
        with open(data_item_dir, 'rb') as file_object:
            data = data + pickle.load(file_object)
    return data

class DataGenerator():
    '''训练数据生成'''
    def __init__(self, opt, items,is_train = True):
        #self.is_training = opt.is_Train
        
        self.is_train = is_train
        self.data = load_data(items,self.is_train)
        random.shuffle(self.data)
        self.batch_size = opt.batchSize
        self.data_nums = len(self.data)
        self.count = 0
        self.ground_truth = []
        self.simulate_image0 = []
        self.simulate_image1 = []
        self.simulate_image2 = []
        self.simulate_image3 = []
        self.filename = []
        for i in range(self.data_nums):
            self.ground_truth.append(self.data[i].ground_truth)
            self.simulate_image0.append(self.data[i].simulate_image0)
            self.simulate_image1.append(self.data[i].simulate_image1)
            self.simulate_image2.append(self.data[i].simulate_image2)
            self.simulate_image3.append(self.data[i].simulate_image3)
            self.filename.append(self.data[i].filename)
    #得到batch数据
    def forward(self,opt):
        train_data_transforms = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomResizedCrop(512,scale=(0.5,1)),
        transforms.ToTensor(),
        ])
        test_data_transforms = transforms.Compose([
        transforms.ToTensor(),
        ])
        self.simulate_image00 = []
        self.simulate_image10 = []
        self.simulate_image20 = []
        self.simulate_image30 = []
        self.ground_truth0 = []
        simulate_image_batch = [[],[],[],[]]
        ground_trunth_batch = []
        filename_batch = []
        number = np.random.randint(314159)
        if self.count + self.batch_size > self.data_nums and self.count < self.data_nums:
            for i in range(self.count,self.data_nums):
                if self.is_train:
                    PIL_image = Image.fromarray(np.uint8(self.simulate_image0[i]*255)) # 这里ndarray_image为原来的numpy数组类型的输入
                    seed_torch(seed = number)
                    self.simulate_image00 = train_data_transforms(PIL_image)
 
                    PIL_image = Image.fromarray(np.uint8(self.simulate_image1[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                    seed_torch(seed = number)
                    self.simulate_image10 = train_data_transforms(PIL_image)
                    
                    PIL_image = Image.fromarray(np.uint8(self.simulate_image2[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                    seed_torch(seed = number)
                    self.simulate_image20 = train_data_transforms(PIL_image)
                    PIL_image = Image.fromarray(np.uint8(self.simulate_image3[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                    seed_torch(seed = number)
                    self.simulate_image30 = train_data_transforms(PIL_image)
                    PIL_image = Image.fromarray(np.uint8(self.ground_truth[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                    seed_torch(seed = number)
                    self.ground_truth0= train_data_transforms(PIL_image)
                else: 
                    PIL_image = Image.fromarray(np.uint8(self.simulate_image0[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                    seed_torch(seed = number)
                    self.simulate_image00 = test_data_transforms(PIL_image)
                    PIL_image = Image.fromarray(np.uint8(self.simulate_image1[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                    seed_torch(seed = number)
                    self.simulate_image10 = test_data_transforms(PIL_image)
                    
                    PIL_image = Image.fromarray(np.uint8(self.simulate_image2[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                    seed_torch(seed = number)
                    self.simulate_image20 = test_data_transforms(PIL_image)                   
                    PIL_image = Image.fromarray(np.uint8(self.simulate_image3[i]*255) ) # 这里ndarray_image为原来的numpy数组类型的输入
                    seed_torch(seed = number)
                    self.simulate_image30 = test_data_transforms(PIL_image)
                    PIL_image = Image.fromarray(np.uint8(self.ground_truth[i]*255) ) # 这里ndarray_image为原来的numpy数组类型的输入
                    seed_torch(seed = number)
                    self.ground_truth0 = test_data_transforms(PIL_image)#array_image为原来的numpy数组类型的输入
                simulate_image_batch[0].append(self.simulate_image00)
                simulate_image_batch[1].append(self.simulate_image10)
                simulate_image_batch[2].append(self.simulate_image20)
                simulate_image_batch[3].append(self.simulate_image30)
                ground_trunth_batch.append(self.ground_truth0)
                filename_batch.append(self.filename[i])
            self.count+=1
        if self.count == self.data_nums:
            self.count = 0
            #print(self.count)
            if self.is_train:
                seed_torch(seed = self.count*number)
                random.shuffle(self.simulate_image0)
                seed_torch(seed = self.count*number)
                random.shuffle(self.simulate_image1)
                seed_torch(seed = self.count*number)
                random.shuffle(self.simulate_image2)
                seed_torch(seed = self.count*number)
                random.shuffle(self.simulate_image3)
                seed_torch(seed = self.count*number)
                random.shuffle(self.ground_truth)
                seed_torch(seed = self.count*number)
                random.shuffle(self.filename)
            else:
                pass
        for i in range(self.count,self.count+self.batch_size):
            
            if self.is_train:
                PIL_image = Image.fromarray(np.uint8(self.simulate_image0[i]*255)) # 这里ndarray_image为原来的numpy数组类型的输入
                seed_torch(seed = self.count*number)
                self.simulate_image00 = train_data_transforms(PIL_image)
                PIL_image = Image.fromarray(np.uint8(self.simulate_image1[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                seed_torch(seed = number)
                self.simulate_image10 = train_data_transforms(PIL_image)
                    
                PIL_image = Image.fromarray(np.uint8(self.simulate_image2[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                seed_torch(seed = number)
                self.simulate_image20 = train_data_transforms(PIL_image)                 
                PIL_image = Image.fromarray(np.uint8(self.simulate_image3[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                seed_torch(seed = self.count*number)
                self.simulate_image30 = train_data_transforms(PIL_image)
                PIL_image = Image.fromarray(np.uint8(self.ground_truth[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                seed_torch(seed = self.count*number)
                self.ground_truth0 = train_data_transforms(PIL_image)
            else: 
                PIL_image = Image.fromarray(np.uint8(self.simulate_image0[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                seed_torch(seed = self.count*number)
                self.simulate_image00 = test_data_transforms(PIL_image)
                PIL_image = Image.fromarray(np.uint8(self.simulate_image1[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                seed_torch(seed = number)
                self.simulate_image10 = test_data_transforms(PIL_image)
                    
                PIL_image = Image.fromarray(np.uint8(self.simulate_image2[i]*255))  # 这里ndarray_image为原来的numpy数组类型的输入
                seed_torch(seed = number)
                self.simulate_image20 = test_data_transforms(PIL_image)                 
                PIL_image = Image.fromarray(np.uint8(self.simulate_image3[i]*255) ) # 这里ndarray_image为原来的numpy数组类型的输入
                seed_torch(seed = self.count*number)
                self.simulate_image30 = test_data_transforms(PIL_image)
                PIL_image = Image.fromarray(np.uint8(self.ground_truth[i]*255) ) # 这里ndarray_image为原来的numpy数组类型的输入
                seed_torch(seed = self.count*number)
                self.ground_truth0 = test_data_transforms(PIL_image)
            simulate_image_batch[0].append(self.simulate_image00)
            simulate_image_batch[1].append(self.simulate_image10)
            simulate_image_batch[2].append(self.simulate_image20)
            simulate_image_batch[3].append(self.simulate_image30)
            ground_trunth_batch.append(self.ground_truth0)
            filename_batch.append(self.filename[i])
        self.count = self.count+self.batch_size
        return ground_trunth_batch, simulate_image_batch,filename_batch
