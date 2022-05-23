import argparse
import os
import random
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

from utils import ToRange255, ToSpaceBGR, \
                  init_patch_square, progress_bar, submatrix

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--target', type=int, default=None, help='')
parser.add_argument('--n-classes', type=int, default=1000, help='')

parser.add_argument('--iter', type=int, default=500, help='Iterations to find adversarial example.')

parser.add_argument('--data', type=str, required=True, help='Input images diretory.')

# TODO: add help msg
parser.add_argument('--x_min', type=int, default=210, help='')
parser.add_argument('--x-max', type=int, default=260, help='')
parser.add_argument('--y-min', type=int, default=210, help='')
parser.add_argument('--y_max', type=int, default=260, help='')

parser.add_argument('--epsilon', type=float, default=5, help='')

parser.add_argument('--image-size', type=int, default=299, help='the height / width of the input image to network')

parser.add_argument('--plot', action='store_true', help='plot all successful adversarial images')

parser.add_argument('--netClassifier', default='inception_v3', 
                    choices=model_names, help="The target classifier")

parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.cuda.set_device(opt.gpu)

# cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

target = opt.target
n_classes = opt.n_classes
# patch_type = opt.patch_type
# patch_size = opt.patch_size
image_size = opt.image_size
plot = opt.plot 
eps = opt.epsilon

if opt.x_min > opt.x_max:
    raise ValueError("x_min > x_max")
if opt.y_min > opt.y_max:
    raise ValueError("y_min > y_max")


print("==> creating model ")
netClassifier = models.__dict__[opt.netClassifier](pretrained=True) # num_classes = 1000 for imagenet
    # pretrainedmodels.__dict__[opt.netClassifier](num_classes=opt.n_classes, pretrained='imagenet')
if opt.cuda:
    netClassifier.cuda()


print('==> Preparing data..')
mean, std = np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]) # TODO: set the dataset's mean and std manually
normalize = transforms.Normalize(mean, std)     

image_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(opt.data, transforms.Compose([
        transforms.Resize(round(image_size*1.050)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # ToSpaceBGR(netClassifier.input_space=='BGR'),         # TODO: set whether need to convert to space BGR manually
        # ToRange255(max(netClassifier.input_range)==255),      # TODO: set whether need to convert to (0, 255) manually
        normalize,
    ])),
    batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=True)
 

min_in, max_in = 0.0, 1.0
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])

min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

def main():
    netClassifier.eval()
    success = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(image_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)

        if target is None:
           targets = torch.randint_like(labels, low=0, high=n_classes) 
        else:
            targets = target * torch.ones_like(labels)


        prediction = netClassifier(data)

        # only computer adversarial examples on examples that are originally classified correctly        
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue
     
        total += 1
        
        data_shape = tuple(data.data.shape)
        patch, mask = init_patch_square(data_shape, opt.x_min, opt.x_max, opt.y_min, opt.y_max)
        if opt.cuda:
            patch = patch.cuda()
            mask = mask.cuda()            

        adv_x, mask, patch = attack(data, patch, mask, labels, targets)

        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        
        if adv_label == target:
            success += 1
      
            if plot: 
                # plot source image
                vutils.save_image(data.data, "./%s/%d_%d_original.png" %(opt.outf, batch_idx, ori_label), normalize=True)
                
                # plot adversarial image
                vutils.save_image(adv_x.data, "./%s/%d_%d_adversarial.png" %(opt.outf, batch_idx, adv_label), normalize=True)
 
        masked_patch = torch.mul(mask, patch)
        # patch = masked_patch.data.cpu().numpy()
        # new_patch = np.zeros_like(patch)
        # for i in range(new_patch.shape[0]): 
        #     for j in range(new_patch.shape[1]): 
        #         new_patch[i][j] = submatrix(patch[i][j])
 
        patch = masked_patch

        # log to file  
        progress_bar(batch_idx, len(image_loader), "Train Patch Success: {:.3f}".format(success/total))

    return patch


def attack(x, patch, mask, source, target):
    netClassifier.eval()

    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
    # src_one_hot = F.one_hot(source).cuda()
    # tar_one_hot = F.one_hot(target).cuda()
    
    for _ in range(1, opt.iter + 1):
        
        adv_x = Variable(adv_x.data, requires_grad=True)
        adv_out = netClassifier(adv_x)

        loss = F.cross_entropy(adv_out, source) -\
               F.cross_entropy(adv_out, target)

        loss.backward()

        adv_grad = adv_x.grad.clone()

        adv_x.grad.data.zero_()

        patch -= adv_grad * eps

        adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
 
    return adv_x, mask, patch 


if __name__ == '__main__':
    print("==> start attack...")
    main()