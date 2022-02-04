
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torch.nn import Softmax
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, norm = 'batch', use_dropout=False, gpu_ids=[], use_parallel=True,
             learn_residual=False):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type = norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    netG = generator(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids,
                         use_parallel=use_parallel, learn_residual=learn_residual)

    # netG = UNet(input_nc)
    netG.apply(weights_init)
    if use_gpu:
        netG.cuda(gpu_ids[0])
    return netG


def define_D(D_input_nc, ndf,n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], use_parallel=True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    netD = NLayerDiscriminator(D_input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Classes
##############################################################################
class Conv_layer(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Conv_layer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            #nn.Tanh(),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv_layers(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv(x)
        ######################################################

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
class external_atttion(nn.Module):
    def __init__(self,in_dim,gf=64):
        super(external_atttion,self).__init__()
        self.conv1 = nn.Conv2d(in_dim,in_dim,1)
        self.mk = nn.Conv1d(in_dim,gf,1,bias=False)
        self.mv=nn.Conv1d(gf,in_dim,1,bias=False)
        self.mv.weight.data = self.mk.weight.data.permute(1,0,2)
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x):
        idn =x
        x = self.conv1(x)
        b,c,h,w=x.size()
        x=x.view(b,c,h*w)
        attn=self.mk(x)
        attn=self.softmax(attn)    
        attn=attn/(0.000000001+(torch.sum(attn,dim=1,keepdim=True)))
        out=self.mv(attn)
        out=out.view(b,c,h,w)
        return out+idn
        
class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim,div):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim//div, kernel_size=1,stride=1, padding=0, padding_mode = 'replicate'),
            nn.BatchNorm2d(in_dim//div),
            )
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim//div, kernel_size=1,stride=1, padding=0, padding_mode = 'replicate'),
            nn.BatchNorm2d(in_dim//div),
            )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1,stride=1, padding=0, padding_mode = 'replicate'),
            nn.BatchNorm2d(in_dim),
            )
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, div = 2):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        CCSatention = CrissCrossAttention(outer_nc,div)
        EXatention = external_atttion(outer_nc,64)
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias, padding_mode = 'replicate')
        conv = nn.Conv2d(outer_nc, outer_nc,kernel_size=3,stride=1,padding=1,padding_mode = 'replicate')
        in_norm = norm_layer(inner_nc)
        ini_norm = norm_layer(outer_nc)
        outt_norm = norm_layer(12)
        prelu = nn.ReLU()
        softmax = nn.Softmax(dim=1)
        out_norm = norm_layer(outer_nc)
        Upsample = nn.Upsample(scale_factor = 2, mode='nearest') 
        conv_upsample = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')

        if outermost:
            conv2 = nn.Conv2d(outer_nc , inner_nc, kernel_size=4, stride=2, padding=1, padding_mode = 'replicate')
            outconv = nn.Conv2d(inner_nc , 12, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
            conv_up = nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
            conv_up2 = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
            down = [conv, prelu, conv2] #第一层
            up = [prelu, Upsample, conv_up, prelu, outconv, outt_norm, prelu]
            model = down + [submodule] + up
        elif innermost:
            conv_up = nn.Conv2d(inner_nc , outer_nc, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
            flat = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
            down = [prelu, CCSatention, CCSatention, conv, prelu, downconv, in_norm]  #最后一层
            up = [prelu, Upsample, conv_up, ini_norm]
            Flat = [prelu, flat, in_norm, prelu, flat, out_norm]
            model = down+up
        else:

            down = [prelu,conv, downconv, in_norm] #中间层
            up = [prelu, Upsample, conv_upsample, ini_norm]


            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up


        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)
######################################################
class UnetGenerator(nn.Module):
    def __init__(self,in_channels,out_channels = 12,norm_layer=nn.BatchNorm2d,innermost = False,outermost = False, use_dropout=False, gpu_ids=[],use_parallel=True,learn_residual=False):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        # currently support only input_nc == output_nc
        #assert(in_channels == out_channels)
        self.conv_layer = Conv_layer(3,32)
        self.conv_layer1 = Conv_layer(64,6)
        self.out_conv = OutConv(12,64) 
        self.out_conv1 = OutConv(64,3) 
        self.out_conv2 = OutConv(15,64) 
        self.out_conv3 = OutConv(79,256) 
        self.out_conv4 = OutConv(335,128) 
        self.out_conv5 = OutConv(463,3) 
        ngf = 64
        self.softmax = nn.Softmax(dim=1)
        # construct unet structure

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, innermost=True, div = 8)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer, div = 8)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer, div = 8)

        unet_block = UnetSkipConnectionBlock(out_channels, ngf, unet_block, outermost=True, norm_layer=norm_layer, div = 2)

        self.model = unet_block
                                   

    def forward(self, inp):
           
            out = self.model(inp)
            output = self.out_conv(out)
            ratio = self.softmax(output)
            output = self.out_conv1(ratio * output)
            f = torch.cat([output, inp], 1)
            f_1 = self.out_conv2(f)
            f_1 = torch.cat([f_1,f],1)
            f_2 = self.out_conv3(f_1)
            f_2 = torch.cat([f_2,f_1],1)
            f_3 = self.out_conv4(f_2)
            f_3 = torch.cat([f_3,f_2],1)
            f_4 = self.out_conv5(f_3)
            
        # else:
        #     output = self.model(z)
        #     output = self.out_conv(output)
        # if self.learn_residual:
        #     output = input3 + output
        #     output = self.out_conv(output)
            return f_4
    ######################################################
    # Defines the PatchGAN discriminator with the specified arguments.
class generator(nn.Module):
    def __init__(self,in_channels,out_channels = 12,norm_layer=nn.BatchNorm2d,innermost = False,outermost = False, use_dropout=False, gpu_ids=[],use_parallel=True,learn_residual=False):
        super(generator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.model = UnetGenerator(in_channels,out_channels = 12,norm_layer=nn.BatchNorm2d,innermost = False,outermost = False, use_dropout=False, gpu_ids=self.gpu_ids,use_parallel=True,learn_residual=False)
    def forward(self, input0, input1, input2, input3):
            inp = torch.cat([input0, input1, input2, input3], 1)
            if self.gpu_ids and isinstance(inp.data, torch.cuda.FloatTensor) and self.use_parallel:
                output = nn.parallel.data_parallel(self.model, inp, self.gpu_ids)
            return output
class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], use_parallel = True):
		super(NLayerDiscriminator, self).__init__()
		self.gpu_ids = gpu_ids
		self.use_parallel = use_parallel
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		kw = 4
		padw = int(np.ceil((kw-1)/2))
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.2, True)
		]

		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2**n, 3)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
						  kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2**n_layers, 3)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
					  kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

		if use_sigmoid:
			sequence += [nn.Sigmoid()]

		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)

######################################################

