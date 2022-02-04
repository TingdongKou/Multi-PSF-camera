from . import pytorch_ssim
import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
###############################################################################
# Functions
###############################################################################


class PerceptualLoss():
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        mean = np.array([.485, .456, .406])
        self.mean = torch.FloatTensor(mean[None, :, None, None]).cuda()
        std = np.array([.229, .224, .225])
        self.std = torch.FloatTensor(std[None, :, None, None]).cuda()
        #self.ssim = pytorch_ssim.SSIM()
        self.criterion = nn.MSELoss()
        self.contentFunc = self.contentFunc()
    def contentFunc(self):
        conv_3_2_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            print(str(i), layer)
            model.add_module(str(i), layer)
            if i == conv_3_2_layer:
                break
        return model

        
        

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)

    def normalize(self, x):
        x = x - self.mean
        x = x / self.std
        return x

    def get_loss(self, fakeIm, realIm):
        realIm = realIm.detach()
        #ssim_loss = 0.1 * (1 - self.ssim(fakeIm,realIm))

        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        G_real = self.gram_matrix(f_real).detach()
        G_fake = self.gram_matrix(f_fake)
        loss = self.criterion(G_fake, G_real) + loss
        loss =  0.1*loss
        print('loss, mse_loss', loss)
        return loss
class RGB_loss(nn.Module):
    def __init__(self):
        super(RGB_loss, self).__init__()
        self.criterion = nn.MSELoss()
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, fusion, Visible1, Visible2, Visible3, Visible4):
        V1_no_grad = Visible1.detach()
        V2_no_grad = Visible2.detach()
        V3_no_grad = Visible3.detach()
        V4_no_grad = Visible4.detach()
        V1_ratio = self.softmax(V1_no_grad)
        V2_ratio = self.softmax(V2_no_grad)
        V3_ratio = self.softmax(V3_no_grad)
        V4_ratio = self.softmax(V4_no_grad)
        F_ratio = self.softmax(fusion)
        rgb_loss = 0.25 * (self.criterion(F_ratio, V1_ratio) + self.criterion(F_ratio, V2_ratio) + self.criterion(F_ratio, V3_ratio) + self.criterion(F_ratio, V4_ratio))
        print('rgb_loss',rgb_loss)
        return rgb_loss    
# class vgg19_out(nn.Module):
#     def __init__(self,requires_grad = False):
#         super(vgg19_out, self).__init__()
        
#         vgg = models.vgg19(pretrained=False).features
#         #vgg.load_state_dict(torch.load('./vgg19-dcbb9e9d.pth'))
#         vgg.cuda()
#         vgg.eval()
#         vgg_pretrained_features = vgg
#         for param in vgg.parameters():
#             param.requires_grad = False
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         for x in range(4):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(4,9):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(9,14):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(14,23):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(23,32):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#     def forward(self, X):
#         h_relu1 = self.slice1(X)
#         h_relu2 = self.slice2(h_relu1)
#         h_relu3 = self.slice3(h_relu2)
#         h_relu4 = self.slice4(h_relu3)
#         h_relu5 = self.slice5(h_relu4)
#         out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
#         return out
# class Perceptual_loss134(nn.Module):
#     def __init__(self):
#         super(Perceptual_loss134, self).__init__()
#         self.vgg = vgg19_out()
#         self.ssim = pytorch_ssim.SSIM()
#         self.criterion = nn.MSELoss()
#         self.weigts = [1.0, 1.0, 1.0, 1.0, 1.0]
#     def forward(self, x, y):
#         y = y.detach()
#         ssim_loss = 0.1 * (1 - self.ssim(x,y))
#         x_vgg = self.vgg(x)
#         y_vgg = self.vgg(y)
#         loss = self.weigts[0]*self.criterion(x_vgg[0], y_vgg[0].detach()) + self.weigts[1]*self.criterion(x_vgg[1], y_vgg[1].detach()) + self.weigts[2]*self.criterion(x_vgg[2], y_vgg[2].detach()) + self.weigts[3]*self.criterion(x_vgg[3], y_vgg[3].detach()) + self.weigts[4]*self.criterion(x_vgg[4], y_vgg[4].detach()) + ssim_loss
#         print(loss)
#         print(ssim_loss)
#         return loss 
    
class GANLoss(nn.Module):
    def __init__(self, use_l1=False, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.target_tensor = None
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = (self.real_label_var is None) or (self.real_label_var.numel() != input.numel())
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            self.target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel())
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                self.target_tensor = self.fake_label_var
        return self.target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, self.target_tensor)
class ESRGanLoss(nn.Module):
    def __init__(self,use_BCE = False):
        super(ESRGanLoss, self).__init__()
    def initialize(self, opt,tensor = torch.FloatTensor):
        self.criterionGAN = nn.BCEWithLogitsLoss()
    def get_g_loss(self, net, fakeB, realB, Tensor = torch.FloatTensor):
        self.pred_real = net.forward(realB.detach())
        self.pred_real = self.pred_real.detach()
        self.pred_fake = net.forward(fakeB)
        self.valid = Variable(Tensor(np.ones((self.pred_real.shape))),requires_grad = False).cuda()
        self.fake = Variable(Tensor(np.zeros((self.pred_fake.shape))),requires_grad = False).cuda()
        return (self.criterionGAN(self.pred_real.mean(0,keepdim=True) - self.pred_fake,self.fake) + self.criterionGAN(self.pred_real - self.pred_fake.mean(0,keepdim=True),self.valid)) * 0.5
    def get_loss(self,net, fakeB, realB, Tensor = torch.FloatTensor):
        self.pred_real = net.forward(realB.detach())
        self.pred_real = self.pred_real.detach()
        self.pred_fake = net.forward(fakeB)
        self.valid = Variable(Tensor(np.ones((self.pred_real.shape))),requires_grad = False).cuda()
        #self.valid = self.valid.to(device=torch.device("cuda:2"))
        self.fake = Variable(Tensor(np.zeros((self.pred_fake.shape))),requires_grad = False).cuda()
        #self.fake = self.valid.to(device=torch.device("cuda:2"))
        self.loss_D_real = self.criterionGAN(self.pred_fake.mean(0,keepdim=True) - self.pred_real, self.valid)      
        self.loss_D_fake = self.criterionGAN(self.pred_fake - self.pred_real.mean(0,keepdim=True), self.fake)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D
class DiscLoss():
    def name(self):
        return 'DiscLoss'

    def initialize(self, opt, tensor):
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)


    def get_g_loss(self,net, fakeB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_fake = net.forward(fakeB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

        # Real
        self.pred_real = net.forward(realB)
        self.loss_D_real = self.criterionGAN(self.pred_real, 1)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D


class DiscLossWGANGP(DiscLoss):
    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.LAMBDA = 10

    def get_g_loss(self, net, fakeB):
        # First, G(A) should fake the discriminator
        self.D_fake = net.forward(fakeB)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(), create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, fakeB, realB):
        self.D_fake = net.forward(fakeB.detach())
        self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(realB)
        self.D_real = self.D_real.mean()
        # Combined loss
        #self.loss_D = self.D_fake - self.D_real

        ##new D loss
        self.one =  y = torch.ones_like(self.D_fake).cuda()
        self.loss_D = (self.D_fake-self.one)**2 + (self.D_real)**2

        gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
        return self.loss_D + gradient_penalty
def init_loss(opt, tensor):
    disc_loss = None
    content_loss = None

    if opt.model == 'content_gan':
        content_loss = PerceptualLoss() 
        rgb_loss = RGB_loss()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    if opt.gan_type == 'wgan-gp':
        disc_loss = DiscLossWGANGP() #
    elif opt.gan_type == 'gan':
        disc_loss = DiscLoss()
    elif opt.gan_type == 'esrgan':
        disc_loss = ESRGanLoss()
    else:
        raise ValueError("GAN [%s] not recognized." % opt.gan_type)
    disc_loss.initialize(opt, tensor)
    return disc_loss, content_loss, rgb_loss
