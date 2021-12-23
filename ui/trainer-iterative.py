from time import time
import os

import torch
os.sys.path.append("..") 
from model_backup import G,D,weights_init

# use pre-defined options.
class opt:
    @property
    def dataset(self):
        return "dataset"
    @property
    def train_batch_size(self):
        return 1
    @property
    def test_batch_size(self):
        return 1
    @property
    def n_epoch(self):
        return 1 #
    @property
    def n_channel_input(self):
        return 4
    @property
    def n_channel_output(self):
        return 4
    @property
    def n_generator_filters(self):
        return 64
    @property
    def n_discriminator_filters(self):
        return 64
    @property
    def lr(self):
        return 0.0002
    @property
    def beta1(self):
        return 0.5
    @property
    def cuda(self):
        return True
    @property
    def resume_G(self):
        return "checkpoint/G"
    @property
    def resume_D(self):
        return "checkpoint/D"
    @property
    def workers(self):
        return 4
    @property
    def seed(self):
        return 123
    @property
    def lamda(self):
        return 10

# override the options.
opt = opt()

# checkpointdir = 'checkpoint'
# prev_G_path = 'default_G.pth'
# prev_D_path = 'default_D.pth'

# N_CHANNEL_INPUT = 4
# N_CHANNEL_OUTPUT = 4
# N_GENERATOR_FILTERS = 64
# N_DISCRIMINATOR_FILTERS = 64

# def load_model():
#     G_filelist = os.listdir(os.path.join(checkpointdir, "G"))
#     D_filelist = os.listdir(os.path.join(checkpointdir, "D"))
#     G_filelist.sort()
#     D_filelist.sort()

#     global netG, netD
#     netG = G(N_CHANNEL_INPUT * 4, N_CHANNEL_OUTPUT, N_GENERATOR_FILTERS)
#     netD = D(N_CHANNEL_INPUT * 4, N_CHANNEL_OUTPUT, N_DISCRIMINATOR_FILTERS)

#     if len(G_filelist) == 0:
#         netG.apply(weights_init)
#     else:
#         netG.load_state_dict(torch.load(os.path.join(checkpointdir, "G", G_filelist[-1])))

#     if len(D_filelist) == 0:
#         netD.apply(weights_init)
#     else:
#         netD.load_state_dict(torch.load(os.path.join(checkpointdir, "D", D_filelist[-1])))

def train_epoch():
    import train    # run the training process.

if __name__ == "__main__":
    while(True):
        train_epoch()
        time.sleep(10)