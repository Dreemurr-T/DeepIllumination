import os
from os.path import join
import argparse
from math import e, log10
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data import DataLoaderHelper

from torch.utils.data import DataLoader
from torch.autograd import Variable
from model_backup import G, D, weights_init
from util import load_image, save_image
# from skimage.measure import compare_ssim as ssim
# from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

checkpointdir = 'checkpoint'

parser = argparse.ArgumentParser(description='DeepRendering-implemention')
parser.add_argument('--dataset', default=".", help='output from unity')
parser.add_argument('--train_batch_size', type=int,
                    default=1, help='batch size for training')
parser.add_argument('--test_batch_size', type=int,
                    default=1, help='batch size for testing')
parser.add_argument('--n_epoch', type=int, default=25,
                    help='number of iterations')
parser.add_argument('--n_channel_input', type=int,
                    default=4, help='number of input channels')
parser.add_argument('--n_channel_output', type=int,
                    default=4, help='number of output channels')
parser.add_argument('--n_generator_filters', type=int,
                    default=64, help='number of initial generator filters')
parser.add_argument('--n_discriminator_filters', type=int,
                    default=64, help='number of initial discriminator filters')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
parser.add_argument('--cuda', action='store_true', help='cuda')
parser.add_argument('--resume_G', help='resume G')
parser.add_argument('--resume_D', help='resume D')
parser.add_argument('--workers', type=int, default=2,
                    help='number of threads for data loader')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--lamda', type=int, default=10,
                    help='L1 regularization factor')

opt = parser.parse_args()

cudnn.benchmark = True

torch.cuda.manual_seed(opt.seed)

batch_size = opt.train_batch_size
n_epoch = opt.n_epoch

print('=> Building model')

netG = G(opt.n_channel_input*4, opt.n_channel_output, opt.n_generator_filters)
netG.apply(weights_init)
netD = D(opt.n_channel_input*4, opt.n_channel_output,
         opt.n_discriminator_filters)
netD.apply(weights_init)

criterion = nn.BCELoss()
criterion_l1 = nn.L1Loss()


albedo = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)
direct = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)
normal = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)
depth = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)

gt = torch.FloatTensor(opt.train_batch_size, opt.n_channel_output, 256, 256)

label = torch.FloatTensor(opt.train_batch_size)
real_label = 1
fake_label = 0

netD = netD.cuda()
netG = netG.cuda()
criterion = criterion.cuda()
criterion_l1 = criterion_l1.cuda()


albedo = albedo.cuda()
direct = direct.cuda()
normal = normal.cuda()
depth = depth.cuda()
gt = gt.cuda()
label = label.cuda()

albedo = Variable(albedo)
direct = Variable(direct)
normal = Variable(normal)
depth = Variable(depth)
gt = Variable(gt)
label = Variable(label)

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

G_filelist = os.listdir(checkpointdir)
G_filelist = [x for x in G_filelist if x.startswith('netG')]
G_filelist.sort()

if len(G_filelist)>0:
    opt.resume_G = os.path.join(checkpointdir, G_filelist[-1])
    lastEpoch = int(opt.resume_G.split("_")[3].split(".")[0])
    opt.resume_D = os.path.join(checkpointdir, "netD_model_epoch_"+str(lastEpoch) + ".pth")
else:
    lastEpoch = 0

if opt.resume_G:
    if os.path.isfile(opt.resume_G):
        print("=> loading generator checkpoint '{}'".format(opt.resume_G))
        checkpoint = torch.load(opt.resume_G)
        lastEpoch = checkpoint['epoch']
        n_epoch = n_epoch - lastEpoch
        netG.load_state_dict(checkpoint['state_dict_G'])
        optimizerG.load_state_dict(checkpoint['optimizer_G'])
        print("=> loaded generator checkpoint '{}' (epoch {})".format(
            opt.resume_G, checkpoint['epoch']))

    else:
        print("=> no checkpoint found")

if opt.resume_D:
    if os.path.isfile(opt.resume_D):
        print("=> loading discriminator checkpoint '{}'".format(opt.resume_D))
        checkpoint = torch.load(opt.resume_D)
        netD.load_state_dict(checkpoint['state_dict_D'])
        optimizerD.load_state_dict(checkpoint['optimizer_D'])
        print("=> loaded discriminator checkpoint '{}'".format(opt.resume_D))


def restore_image(image):
    image = image.add_(1).div_(2)
    image *= 255.0
    image = image.clip(0, 255)
    return image

root_dir = "dataset/"
train_dir = join(root_dir + opt.dataset, "train")
test_dir = join(root_dir + opt.dataset, "val")

def load_training_dataset():
    global train_set, train_data
    print('=> Loading training datasets')
    train_set = DataLoaderHelper(train_dir)
    train_data = DataLoader(dataset=train_set, num_workers=opt.workers,
                            batch_size=opt.train_batch_size, shuffle=True)

load_training_dataset()

def train(epoch):

    for (i, images) in enumerate(train_data):
        netD.zero_grad()
        (albedo_cpu, direct_cpu, normal_cpu, depth_cpu, gt_cpu) = (
            images[0], images[1], images[2], images[3], images[4])
        # albedo.data.resize_(albedo_cpu.size()).copy_(albedo_cpu)
        # direct.data.resize_(direct_cpu.size()).copy_(direct_cpu)
        # normal.data.resize_(normal_cpu.size()).copy_(normal_cpu)
        # depth.data.resize_(depth_cpu.size()).copy_(depth_cpu)
        # gt.data.resize_(gt_cpu.size()).copy_(gt_cpu)
        # print(albedo.size())
        # print(albedo_cpu.size())
        albedo.resize_(albedo_cpu.size()).copy_(albedo_cpu)
        direct.resize_(direct_cpu.size()).copy_(direct_cpu)
        normal.resize_(normal_cpu.size()).copy_(normal_cpu)
        depth.resize_(depth_cpu.size()).copy_(depth_cpu)
        gt.resize_(gt_cpu.size()).copy_(gt_cpu)
        # print(albedo.size())
        # print(albedo_cpu.size())
        output = netD(torch.cat((albedo, direct, normal, depth, gt), 1))
        # label.data.resize_(output.size()).fill_(real_label)
        label.resize_(output.size()).fill_(real_label)
        err_d_real = criterion(output, label)
        err_d_real.backward()
        d_x_y = output.data.mean()
        fake_B = netG(torch.cat((albedo, direct, normal, depth), 1))
        output = netD(
            torch.cat((albedo, direct, normal, depth, fake_B.detach()), 1))
        # label.data.resize_(output.size()).fill_(fake_label)
        label.resize_(output.size()).fill_(fake_label)
        err_d_fake = criterion(output, label)
        err_d_fake.backward()
        d_x_gx = output.data.mean()
        err_d = (err_d_real + err_d_fake) * 0.5
        optimizerD.step()

        netG.zero_grad()
        output = netD(torch.cat((albedo, direct, normal, depth, fake_B), 1))
        # label.data.resize_(output.size()).fill_(real_label)
        label.resize_(output.size()).fill_(real_label)
        ssim_loss = SSIM(win_size=11, win_sigma=1.5,
                         data_range=1, size_average=True, channel=4)
        _ssim_loss = 1-ssim_loss(fake_B, gt)
        # print(_ssim_loss.item())
        err_g = criterion(output, label) + opt.lamda \
            * 0.3*criterion_l1(fake_B, gt) + opt.lamda * 0.7 * _ssim_loss
        err_g.backward()
        d_x_gx_2 = output.data.mean()
        optimizerG.step()
        print('=> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}'.format(
            epoch,
            i,
            len(train_data),
            err_d.item(),
            err_g.item(),
            d_x_y,
            d_x_gx,
            d_x_gx_2,
        ))


def save_checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(
        opt.dataset, epoch)
    net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(
        opt.dataset, epoch)
    torch.save({'epoch': epoch+1, 'state_dict_G': netG.state_dict(),
               'optimizer_G': optimizerG.state_dict()}, net_g_model_out_path)
    torch.save({'state_dict_D': netD.state_dict(),
               'optimizer_D': optimizerD.state_dict()}, net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

    if not os.path.exists("validation"):
        os.mkdir("validation")
    if not os.path.exists(os.path.join("validation", opt.dataset)):
        os.mkdir(os.path.join("validation", opt.dataset))

    if epoch == n_epoch:

        print('=> Loading validation datasets')

        val_set = DataLoaderHelper(test_dir)

        val_data = DataLoader(dataset=val_set, num_workers=opt.workers,
                    batch_size=opt.test_batch_size, shuffle=False)

        for index, images in enumerate(val_data):
            (albedo_cpu, direct_cpu, normal_cpu, depth_cpu, gt_cpu) = (
                images[0], images[1], images[2], images[3], images[4])
    #         # albedo.data.resize_(albedo_cpu.size()).copy_(albedo_cpu)
    #         # direct.data.resize_(direct_cpu.size()).copy_(direct_cpu)
    #         # normal.data.resize_(normal_cpu.size()).copy_(normal_cpu)
    #         # depth.data.resize_(depth_cpu.size()).copy_(depth_cpu)
            albedo.resize_(albedo_cpu.size()).copy_(albedo_cpu)
            direct.resize_(direct_cpu.size()).copy_(direct_cpu)
            normal.resize_(normal_cpu.size()).copy_(normal_cpu)
            depth.resize_(depth_cpu.size()).copy_(depth_cpu)
            out = netG(torch.cat((albedo, direct, normal, depth), 1))
            out = out.cpu()
            out_img = out.data[0]
            save_image(
                out_img, "validation/{}/{}_Fake.png".format(opt.dataset, index))
            save_image(
                gt_cpu[0], "validation/{}/{}_Real.png".format(opt.dataset, index))
            save_image(
                direct_cpu[0], "validation/{}/{}_Direct.png".format(opt.dataset, index))


if __name__ == '__main__':
    for epoch in tqdm(range(n_epoch)):
        train(epoch+lastEpoch)
        if epoch % 1 == 0:
            save_checkpoint(epoch+lastEpoch)
