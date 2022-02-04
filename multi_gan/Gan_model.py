import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .losses import init_loss
from . import model
import torch.nn as nn
import matplotlib.pyplot as plt
from util import util
try:
    xrange          # Python2
except NameError:
    xrange = range  # Python 3

class ConditionalGAN(nn.Module):
    def name(self):
        return 'ConditionalGANModel'
    def __init__(self,opt, use_dropout):
        super(ConditionalGAN, self).__init__()
        self.opt = opt
        self.use_dropout = use_dropout
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.isTrain = opt.isTrain
        self.learn_residual = opt.learn_residual
        self.n_layers_D = opt.n_layers_D
        # define tensors
        self.input_A0 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_A1 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_A2 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_A3 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.real = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # load/define model
        # Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
        #use_parallel = not opt.gan_type == 'wgan-gp'
        self.netG = model.define_G(opt.input_nc, opt.output_nc, opt.norm,
                                   self.use_dropout, self.gpu_ids, use_parallel = True, learn_residual=self.learn_residual)
        if self.isTrain:
            use_sigmoid = opt.gan_type == 'gan'
            self.netD = model.define_D(opt.D_input_nc, opt.ndf, self.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel = True)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
        if self.isTrain:
            self.old_lr = opt.lr

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1

            # define loss functions
            self.discLoss, self.contentLoss, self.rgb_loss = init_loss(opt, self.Tensor)

        #print('---------- model initialized -------------')
       # model.print_network(self.netG)
        #if self.isTrain:
           # model.print_network(self.netD)
       # print('-----------------------------------------------')

    def set_input(self, real,blur0,blur1, blur2, blur3):
        self.input_A0.resize_(blur0.size()).copy_(blur0)
        self.input_A1.resize_(blur1.size()).copy_(blur1)
        self.input_A2.resize_(blur2.size()).copy_(blur2)
        self.input_A3.resize_(blur3.size()).copy_(blur3)
        self.real.resize_(real.size()).copy_(real)

    def forward(self):
        self.real_A0 = Variable(self.input_A0)
        self.real_A1 = Variable(self.input_A1)
        self.real_A2 = Variable(self.input_A2)
        self.real_A3 = Variable(self.input_A3)
        self.fake_B = self.netG.forward(self.real_A0, self.real_A1, self.real_A2, self.real_A3)
        self.real_B = Variable(self.real)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A0 = Variable(self.input_A0)
            self.real_A1 = Variable(self.input_A1)
            self.real_A2 = Variable(self.input_A2)
            self.real_A3 = Variable(self.input_A3)
            self.fake_B = self.netG.forward(self.real_A0, self.real_A1, self.real_A2, self.real_A3)
            self.real_B = Variable(self.real)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        self.loss_D = self.discLoss.get_loss(self.netD, self.fake_B, self.real_B)  # WGAN loss

        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.fake_B,self.real_B)  # WGAN LOSS
        # Second, G(A) = B
        self.loss_G_Content = self.contentLoss.get_loss(self.fake_B,self.real_B) * self.opt.lambda_A * 0.1  # Perpectual loss
        self.loss_rgb = self.rgb_loss.forward(self.fake_B, self.real_A0, self.real_A1,self.real_A2,self.real_A3)
        self.loss_G = self.loss_G_GAN + self.loss_G_Content + self.loss_rgb

        self.loss_G.backward(retain_graph=True)
    def validation_backward_D(self):
        self.loss_D = self.discLoss.get_loss(self.netD, self.fake_B, self.real_B)  # WGAN loss

      

    def validation_backward_G(self):
        self.loss_G_GAN = self.discLoss.get_g_loss(self.netD,  self.fake_B, self.real_B)  # WGAN LOSS
        # Second, G(A) = B
        self.loss_G_Content = self.contentLoss.get_loss(self.fake_B,self.real_B) * self.opt.lambda_A * 0.1  # Perpectual loss

        self.loss_rgb = self.rgb_loss.forward(self.fake_B, self.real_A0, self.real_A1,self.real_A2,self.real_A3)
        self.loss_G = self.loss_G_GAN + self.loss_G_Content + self.loss_rgb



    # 训练过程，先训练D,后训练G。
    def train_optimize_parameters(self):
        self.forward()

        for iter_d in xrange(self.criticUpdates):
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def validation_optimize_parameters(self):
        self.test()
        for iter_d in xrange(self.criticUpdates):
            self.optimizer_D.zero_grad()
            self.validation_backward_D()
            #self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.validation_backward_G()
        #self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('ContentLoss', self.loss_G_Content.item()),
                            ('G_Loss', self.loss_G.item()),
                            ('D_Loss', self.loss_D.item())])

    def get_current_visuals(self):
        real_A0,real_A1,real_A2,real_A3,fake_B,real_B,real_A0_1,real_A1_1,real_A2_1,real_A3_1,fake_B_1,real_B_1, = util.tensor2im(self.real_A0.data, self.real_A1.data, self.real_A2.data, self.real_A3.data, self.fake_B.data, self.real_B.data)
        return OrderedDict([('Blurred_Train0', real_A0),('Blurred_Train0_1', real_A0_1), ('Blurred_Train1', real_A1), ('Blurred_Train1_1', real_A1_1), ('Blurred_Train2', real_A2), ('Blurred_Train2_1', real_A2_1),('Blurred_Train3', real_A3), ('Blurred_Train3_1', real_A3_1),('Restored_Train', fake_B), ('Restored_Train_1', fake_B_1),('Sharp_Train', real_B),('Sharp_Train_1', real_B_1)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

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
