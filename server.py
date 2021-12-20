import torch
import argparse
from model import G, D
from util import load_image, save_image

# read the generated checkpoint data from train.py

G_pth_path = 'checkpoint/JiaRan/netG_model_epoch_199.pth'
# D_pth_path = 'netD_model_epoch_199.pth'

parser = argparse.ArgumentParser(description='DeepRendering-server')
parser.add_argument('--n_channel_input', type=int,
                    default=4, help='number of input channels')
parser.add_argument('--n_channel_output', type=int,
                    default=4, help='number of output channels')
parser.add_argument('--n_generator_filters', type=int,
                    default=64, help='number of initial generator filters')
opt = parser.parse_args()

loaded_model = torch.load(G_pth_path, map_location=torch.device('cpu'))
netG = G(opt.n_channel_input * 4, opt.n_channel_output, opt.n_generator_filters)
netG.load_state_dict(loaded_model['state_dict_G'])

albedo_path = 'dataset/nana7mi/albedo/nana7mi0001.png'
direct_path = 'dataset/nana7mi/direct/nana7mi0001.png'
normal_path = 'dataset/nana7mi/normal/nana7mi0001.png'
depth_path = 'dataset/nana7mi/depth/nana7mi0001.png'

albedo_image = load_image(albedo_path)
direct_image = load_image(direct_path)
normal_image = load_image(normal_path)
depth_image = load_image(depth_path)

albedo = torch.autograd.Variable(albedo_image).view(1, -1, 256, 256)
direct = torch.autograd.Variable(direct_image).view(1, -1, 256, 256)
normal = torch.autograd.Variable(normal_image).view(1, -1, 256, 256)
depth = torch.autograd.Variable(depth_image).view(1, -1, 256, 256)

out = netG(torch.cat((albedo, direct, normal, depth), 1))
out_img = out.data[0]

save_image(out_img, 'result.png')