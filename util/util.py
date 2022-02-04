from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import matplotlib.pyplot as plt

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(blur0=[],blur1=[],blur2=[],blur3=[],fake=[],real=[],imtype=np.float32):
    image_numpy_blur0_0 = [[], []]
    image_numpy_blur1_0 = [[], []]
    image_numpy_blur2_0 = [[], []]
    image_numpy_blur3_0 = [[], []]
    image_numpy_fake_0 = [[], []]
    image_numpy_real_0 = [[], []]
    image_numpy_blur0 = blur0.cpu().float().numpy()
    image_numpy_blur0_0 = image_numpy_blur0[0,:,:,:]
    image_numpy_blur0_1 = image_numpy_blur0[1,:,:,:]
    #print(image_numpy_blur0.shape)
    image_numpy_blur0_0 = (np.transpose(image_numpy_blur0_0, (1, 2, 0)))
    image_numpy_blur0_1 = (np.transpose(image_numpy_blur0_1, (1, 2, 0)))
    #plt.imshow(image_numpy_blur0)
    #plt.show()
    image_numpy_blur1 = blur1.cpu().float().numpy()
    image_numpy_blur1_0 = image_numpy_blur1[0,:,:,:]
    image_numpy_blur1_1 = image_numpy_blur1[1,:,:,:]
    image_numpy_blur1_0 = (np.transpose(image_numpy_blur1_0, (1, 2, 0)))
    image_numpy_blur1_1 = (np.transpose(image_numpy_blur1_1, (1, 2, 0)))
    
    image_numpy_blur2 = blur2.cpu().float().numpy()
    image_numpy_blur2_0 = image_numpy_blur2[0,:,:,:]
    image_numpy_blur2_1 = image_numpy_blur2[1,:,:,:]
    image_numpy_blur2_0 = (np.transpose(image_numpy_blur2_0, (1, 2, 0)))
    image_numpy_blur2_1 = (np.transpose(image_numpy_blur2_1, (1, 2, 0)))

    image_numpy_blur3 = blur3.cpu().float().numpy()
    image_numpy_blur3_0 = image_numpy_blur3[0,:,:,:]
    image_numpy_blur3_1 = image_numpy_blur3[1,:,:,:]
    image_numpy_blur3_0 = (np.transpose(image_numpy_blur3_0, (1, 2, 0)))
    image_numpy_blur3_1 = (np.transpose(image_numpy_blur3_1, (1, 2, 0)))

    image_numpy_fake = fake.cpu().float().numpy()
    image_numpy_fake_0 = image_numpy_fake[0,:,:,:]
    image_numpy_fake_1 = image_numpy_fake[1,:,:,:]
    image_numpy_fake_0 = (np.transpose(image_numpy_fake_0, (1, 2, 0)))
    image_numpy_fake_1 = (np.transpose(image_numpy_fake_1, (1, 2, 0)))

    image_numpy_real = real.cpu().float().numpy()
    image_numpy_real_0 = image_numpy_real[0,:,:,:]
    image_numpy_real_1 = image_numpy_real[1,:,:,:]
    image_numpy_real_0 = (np.transpose(image_numpy_real_0, (1, 2, 0)))
    image_numpy_real_1 = (np.transpose(image_numpy_real_1, (1, 2, 0)))

#    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy_blur0_0.astype(imtype), image_numpy_blur1_0.astype(imtype), image_numpy_blur2_0.astype(imtype), image_numpy_blur3_0.astype(imtype),image_numpy_fake_0.astype(imtype), image_numpy_real_0.astype(imtype),image_numpy_blur0_1.astype(imtype), image_numpy_blur1_1.astype(imtype), image_numpy_blur2_1.astype(imtype), image_numpy_blur3_1.astype(imtype),image_numpy_fake_1.astype(imtype), image_numpy_real_1.astype(imtype)
def tensor2im_test_2(blur0=[],blur3=[],fake=[],real=[],imtype=np.float32):
    image_numpy_blur0_0 = [[], []]
 
    image_numpy_blur3_0 = [[], []]
    image_numpy_fake_0 = [[], []]
    image_numpy_real_0 = [[], []]
    image_numpy_blur0 = blur0.cpu().float().numpy()
    image_numpy_blur0_0 = image_numpy_blur0[0,:,:,:]
 
    #print(image_numpy_blur0.shape)
    image_numpy_blur0_0 = (np.transpose(image_numpy_blur0_0, (1, 2, 0)))

  
  

    image_numpy_blur3 = blur3.cpu().float().numpy()
    image_numpy_blur3_0 = image_numpy_blur3[0,:,:,:]
  
    image_numpy_blur3_0 = (np.transpose(image_numpy_blur3_0, (1, 2, 0)))
    

    image_numpy_fake = fake.cpu().float().numpy()
    image_numpy_fake_0 = image_numpy_fake[0,:,:,:]

    image_numpy_fake_0 = (np.transpose(image_numpy_fake_0, (1, 2, 0)))


    image_numpy_real = real.cpu().float().numpy()
    image_numpy_real_0 = image_numpy_real[0,:,:,:]
 
    image_numpy_real_0 = (np.transpose(image_numpy_real_0, (1, 2, 0)))


#    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy_blur0_0.astype(imtype), image_numpy_blur3_0.astype(imtype),image_numpy_fake_0.astype(imtype), image_numpy_real_0.astype(imtype)
def tensor2im_test(blur0=[],blur1=[],blur2=[],blur3=[],fake=[],real=[],imtype=np.float32):
    image_numpy_blur0_0 = [[], []]
    image_numpy_blur1_0 = [[], []]
    image_numpy_blur2_0 = [[], []]
    image_numpy_blur3_0 = [[], []]
    image_numpy_fake_0 = [[], []]
    image_numpy_real_0 = [[], []]
    image_numpy_blur0 = blur0.cpu().float().numpy()
    image_numpy_blur0_0 = image_numpy_blur0[0,:,:,:]
 
    #print(image_numpy_blur0.shape)
    image_numpy_blur0_0 = (np.transpose(image_numpy_blur0_0, (1, 2, 0)))

    #plt.imshow(image_numpy_blur0)
    #plt.show()
    image_numpy_blur1 = blur1.cpu().float().numpy()
    image_numpy_blur1_0 = image_numpy_blur1[0,:,:,:]

    image_numpy_blur1_0 = (np.transpose(image_numpy_blur1_0, (1, 2, 0)))

    
    image_numpy_blur2 = blur2.cpu().float().numpy()
    image_numpy_blur2_0 = image_numpy_blur2[0,:,:,:]
 
    image_numpy_blur2_0 = (np.transpose(image_numpy_blur2_0, (1, 2, 0)))
  

    image_numpy_blur3 = blur3.cpu().float().numpy()
    image_numpy_blur3_0 = image_numpy_blur3[0,:,:,:]
  
    image_numpy_blur3_0 = (np.transpose(image_numpy_blur3_0, (1, 2, 0)))
    

    image_numpy_fake = fake.cpu().float().numpy()
    image_numpy_fake_0 = image_numpy_fake[0,:,:,:]

    image_numpy_fake_0 = (np.transpose(image_numpy_fake_0, (1, 2, 0)))


    image_numpy_real = real.cpu().float().numpy()
    image_numpy_real_0 = image_numpy_real[0,:,:,:]
 
    image_numpy_real_0 = (np.transpose(image_numpy_real_0, (1, 2, 0)))


#    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy_blur0_0.astype(imtype), image_numpy_blur1_0.astype(imtype), image_numpy_blur2_0.astype(imtype), image_numpy_blur3_0.astype(imtype),image_numpy_fake_0.astype(imtype), image_numpy_real_0.astype(imtype)
def tensor2im_2(blur0=[],  blur3=[], fake=[], real=[], imtype=np.float32):

    image_numpy_blur0 = blur0.cpu().float().numpy()
    image_numpy_blur0_0 = image_numpy_blur0[0, :, :, :]
    image_numpy_blur0_1 = image_numpy_blur0[1, :, :, :]
    # print(image_numpy_blur0.shape)
    image_numpy_blur0_0 = (np.transpose(image_numpy_blur0_0, (1, 2, 0)))
    image_numpy_blur0_1 = (np.transpose(image_numpy_blur0_1, (1, 2, 0)))
    # plt.imshow(image_numpy_blur0)
    # plt.show()

    image_numpy_blur3 = blur3.cpu().float().numpy()
    image_numpy_blur3_0 = image_numpy_blur3[0, :, :, :]
    image_numpy_blur3_1 = image_numpy_blur3[1, :, :, :]
    image_numpy_blur3_0 = (np.transpose(image_numpy_blur3_0, (1, 2, 0)))
    image_numpy_blur3_1 = (np.transpose(image_numpy_blur3_1, (1, 2, 0)))

    image_numpy_fake = fake.cpu().float().numpy()
    image_numpy_fake_0 = image_numpy_fake[0, :, :, :]
    image_numpy_fake_1 = image_numpy_fake[1, :, :, :]
    image_numpy_fake_0 = (np.transpose(image_numpy_fake_0, (1, 2, 0)))
    image_numpy_fake_1 = (np.transpose(image_numpy_fake_1, (1, 2, 0)))

    image_numpy_real = real.cpu().float().numpy()
    image_numpy_real_0 = image_numpy_real[0, :, :, :]
    image_numpy_real_1 = image_numpy_real[1, :, :, :]
    image_numpy_real_0 = (np.transpose(image_numpy_real_0, (1, 2, 0)))
    image_numpy_real_1 = (np.transpose(image_numpy_real_1, (1, 2, 0)))

    #    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy_blur0_0.astype(imtype),  image_numpy_blur3_0.astype(imtype), image_numpy_fake_0.astype(imtype), image_numpy_real_0.astype(
        imtype), image_numpy_blur0_1.astype(imtype),  image_numpy_blur3_1.astype(imtype), image_numpy_fake_1.astype(imtype), image_numpy_real_1.astype(
        imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = None
    if image_numpy.shape[2] == 1:
        image_numpy = np.reshape(image_numpy, (image_numpy.shape[0],image_numpy.shape[1]))
        image_pil = Image.fromarray(image_numpy, 'L')
    else:
        image_numpy = image_numpy**(1/1.8)
        image_numpy = image_numpy*255.
        image_numpy = np.uint8(image_numpy)
        image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings. Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" % (method.ljust(spacing), processFunc(str(getattr(object, method).__doc__))) for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
