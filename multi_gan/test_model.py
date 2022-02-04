import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .losses import init_loss
from . import model
import torch.nn as nn
from util import util
class TestModel():
    def name(self):
        return 'TestModel'

    def __init__(self,opt,use_dropout):
        super(TestModel, self).__init__()
        #assert(not opt.isTrain)
        self.opt=opt
        self.gpu_ids = opt.gpu_ids
        self.use_dropout = use_dropout
        self.n_layers_D = opt.n_layers_D
        self.Tensor = torch.cuda.FloatTensor 
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        with torch.no_grad():
            self.input_A0 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
            self.input_A1 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
            self.input_A2 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
            self.input_A3 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
            self.real = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

            self.netG = model.define_G(opt.input_nc, opt.output_nc, opt.norm, self.use_dropout, self.gpu_ids, use_parallel=True, learn_residual=opt.learn_residual)
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            use_sigmoid = opt.gan_type == 'gan'
            self.netD = model.define_D(opt.D_input_nc, opt.ndf, self.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel = True)
            self.load_network(self.netD, 'D', which_epoch)
            self.netG.eval()
            self.netD.eval()
            self.discLoss, self.contentLoss, self.rgb_loss = init_loss(opt, self.Tensor)
        #print('---------- Networks initialized -------------')
        #model.print_network(self.netG)
        #print('-----------------------------------------------')
    def set_input(self, real,blur0,blur1,blur2,blur3):
        # we need to use single_dataset mode
        with torch.no_grad():
            self.input_A0.resize_(blur0.size()).copy_(blur0)
            self.input_A1.resize_(blur1.size()).copy_(blur1)
            self.input_A2.resize_(blur2.size()).copy_(blur2)
            self.input_A3.resize_(blur3.size()).copy_(blur3)
            self.real.resize_(real.size()).copy_(real)

    def test(self):
        with torch.no_grad():
            self.real_A0 = self.input_A0
            self.real_A1 = self.input_A1
            self.real_A2 = self.input_A2
            self.real_A3 = self.input_A3
            self.fake_B = self.netG.forward(self.real_A0, self.real_A1, self.real_A2, self.real_A3).detach()
            self.real_B = self.real
    # get image paths
    def get_image_paths(self):
        return self.image_paths
    def test_backward(self):
        self.loss_G_GAN = self.discLoss.get_g_loss(self.netD,  self.fake_B,self.real_B)  # WGAN LOSS
        # Second, G(A) = B
        self.loss_G_Content = self.contentLoss.get_loss(self.fake_B,self.real_B)                                            # Perpectual loss

        self.loss_rgb = self.rgb_loss.forward(self.fake_B, self.real_A0, self.real_A1,self.real_A2,self.real_A3)
        self.loss_G = self.loss_G_GAN + self.loss_G_Content + self.loss_rgb
    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('ContentLoss', self.loss_G_Content.item()),
                            ('rgb_Loss', self.loss_rgb.item()),
                            ('G_Loss', self.loss_G.item())])
    def get_current_visuals(self):
        real_A0,real_A1,real_A2,real_A3,fake_B,real_B = util.tensor2im_test(self.real_A0.data, self.real_A1.data, self.real_A2.data, self.real_A3.data, self.fake_B.data, self.real_B.data)
        return OrderedDict([('Blurred_Train0', real_A0), ('Blurred_Train1', real_A1), ('Blurred_Train2', real_A2), ('Blurred_Train3', real_A3), ('Restored_Train', fake_B),('Sharp_Train', real_B)])

    ######################################################
    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
    ######################################################
    ######################################################
