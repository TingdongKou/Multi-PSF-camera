# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import scipy.io
import os
import random
import math
import numpy as np
import sys
import glob

class TrainingData():
    def __init__(self, filename, ground_truth, simulate_image0, simulate_image1, simulate_image2, simulate_image3):
        self.filename = filename
        self.ground_truth = ground_truth
        self.simulate_image0 = simulate_image0
        self.simulate_image1 = simulate_image1
        self.simulate_image2 = simulate_image2
        self.simulate_image3 = simulate_image3

class SpectralData():
    '''数据类，导入数据等'''
    def __init__(self, spec_direction):
        # self.spec_names = sorted(os.listdir(spec_direction))
        self.spec_names = glob.glob(os.path.join(spec_direction, '*'))
        self.spectral_datas = []
        self.spec_direction = spec_direction
        print(self.spec_names)
    def load(self, datas):     
        load_count=0
        for name in self.spec_names:
            load_count+=1
            if load_count%1000 == 0:
                print('now we are loading ', load_count, ' data')
            #matlab_data = scipy.io.loadmat(os.path.join(self.spec_direction, name))
            matlab_data = scipy.io.loadmat(name)
            filename = matlab_data['filename']
            ground_truth = matlab_data['ground_truth']
            simulate_image0 = matlab_data['simulate_image0']
            simulate_image1 = matlab_data['simulate_image1']
            simulate_image2 = matlab_data['simulate_image2']
            simulate_image3 = matlab_data['simulate_image3']
            simulate_image0 = np.clip(simulate_image0 / simulate_image0.max(), 0, 1)
            simulate_image1 = np.clip(simulate_image1 / simulate_image1.max(), 0, 1)
            simulate_image2 = np.clip(simulate_image2 / simulate_image2.max(), 0, 1)
            simulate_image3 = np.clip(simulate_image3 / simulate_image3.max(), 0, 1)
            simulate_image0 = simulate_image0.astype(np.float32)
            simulate_image1 = simulate_image1.astype(np.float32)
            simulate_image2 = simulate_image2.astype(np.float32)
            simulate_image3 = simulate_image3.astype(np.float32)         
            ground_truth = np.clip(ground_truth / ground_truth.max(), 0, 1)
            ground_truth = ground_truth.astype(np.float32)            
            item = TrainingData(filename[0], ground_truth, simulate_image0, simulate_image1, simulate_image2, simulate_image3)
            datas.append(item)
            
def cross_split(datas, slice_nums):
    datas_cross = [[][:] for index in range(slice_nums)]
    random.shuffle(datas)
    number = math.floor(len(datas)/slice_nums)
    for i in range(slice_nums):
        print('\n we are spliting ', i, ' datas')
        start = i * number
        end = min(start + number, len(datas))
        for j in range(start, end):
            datas_cross[i].append(datas[j])
            print('the ', j , ' filename is', datas[j].filename)
            
    return datas_cross
    
def data_dump(datas_cross, record_dir):
    with open(record_dir, 'wb') as file_object:
        pickle.dump(datas_cross, file_object, protocol=-1)    
    
    
def train_make():
    spec_dir = './data_input/train_matdata/white_data/'
    record_dir = './data_input/train_pkldata/pkldata'
    training_datas = []
    spectral_data = SpectralData(spec_dir)
    spectral_data.load(training_datas)
    training_datas_cross = cross_split(training_datas, 10)
    print('data was dumped now')
    for item, cross_data in enumerate(training_datas_cross):
        record_dir_item = record_dir + str(item) + '.pkl'
        print('the ',item + 1,' of', len(training_datas_cross), ' training datas is', len(cross_data))        
        data_dump(cross_data, record_dir_item)
    print('dump over')

def test_make():
    spec_dir = './data_input/test_matdata/'
    record_dir = './data_input/test_pkldata/pkldata'
    test_datas = []
    spectral_data = SpectralData(spec_dir)
    spectral_data.load(test_datas)
    test_datas_cross = cross_split(test_datas, 1)
    print('data was dumped now')
    for item, cross_data in enumerate(test_datas_cross):
        record_dir_item = record_dir + '99.pkl'
        print('the ',item + 1,' of', len(test_datas_cross), ' training datas is', len(cross_data))            
        data_dump(cross_data, record_dir_item)    
    print('dump over')
    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please tell me more about the function you want, such as train_make or test_make')
        exit(-1)
    function = sys.argv[1]
    globals()[function](*sys.argv[2:])

    
    
