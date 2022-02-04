import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from multi_gan.losses import init_loss
from multi_gan import model
from multi_gan.test_model import TestModel
import scipy.io
import torch.nn as nn
import time
from options.test_options import TestOptions
#from util.visualizer import Visualizer
from predata.data_generator import DataGenerator
from predata import datasets_processing
import torch.utils.data as Data
from util.metrics import PSNR, SSIM
from util import util
def test():
    opt = TestOptions().parse()
    datasets_processing.test_make()
    test_data_loader = DataGenerator(opt, opt.test_cross_items,opt.isTrain)
    model = TestModel(opt,use_dropout=False)
    errors_all =0
    pl_all =0
    batch_nums = test_data_loader.data_nums // opt.batchSize
    psnr_all=0
    ssim_all=0
    print(batch_nums)
    
    for j in range(batch_nums):
        start_time = time.time()
        real, blur,filename = test_data_loader.forward(opt)
        blur0_0 = blur[0]
        blur0_1 = blur[1]
        blur0_2 = blur[2]
        blur0_3 = blur[3]
        for i in range(opt.batchSize):
                real[i] = torch.unsqueeze(real[i], 0)
                blur0_0[i] = torch.unsqueeze(blur0_0[i], 0)
                blur0_1[i] = torch.unsqueeze(blur0_1[i], 0)
                blur0_2[i] = torch.unsqueeze(blur0_2[i], 0)
                blur0_3[i] = torch.unsqueeze(blur0_3[i], 0)
        real = torch.cat([real[i] for i in range(opt.batchSize)],0)
        blur0 = torch.cat([blur0_0[i] for i in range(opt.batchSize)],0)
        blur1 = torch.cat([blur0_1[i] for i in range(opt.batchSize)],0)
        blur2 = torch.cat([blur0_2[i] for i in range(opt.batchSize)],0)
        blur3 = torch.cat([blur0_3[i] for i in range(opt.batchSize)],0)
        blur_size = len(blur0)
        print('###training blur_images = %d' % blur_size)
        model.set_input(real, blur0, blur1, blur2, blur3)
        #print(real.shape)
        model.test()
        print(time.time()-start_time)
        print(filename[0])
        model.test_backward()
        results = model.get_current_visuals()
        errors = model.get_current_errors()
        psnrMetric = PSNR(results['Restored_Train'], results['Sharp_Train'])
        #psnrMetric_1 = PSNR(results['Restored_Train_1'], results['Sharp_Train_1'])
        psnr_all = psnr_all + psnrMetric 
        psnr_avg = psnr_all/(batch_nums)
        scipy.io.savemat('./data_output/test_CCAR_IN2/loss_' + str(j) + '.mat',
                         {'G_GAN':errors['G_GAN'], 'perceptual': errors['ContentLoss'], 'G_Loss': errors['G_Loss'] })
        errors_all = errors_all + errors['G_Loss']
        pl_all = pl_all + errors['ContentLoss']
        scipy.io.savemat('./data_output/test_CCAR_IN2/batch_' + str(j) + '.mat',
                         {'Blurred_Train0':results['Blurred_Train0'], 'Blurred_Train1':results['Blurred_Train1'], 'Blurred_Train2':results['Blurred_Train2'],'Blurred_Train3':results['Blurred_Train3'],'restored_image': results['Restored_Train'], 'sharp_train': results['Sharp_Train'], 'filename_0': filename[0]})
    print(psnr_avg)
    print(errors_all)
    print(pl_all)
    
test()
