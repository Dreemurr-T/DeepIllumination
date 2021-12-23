import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, n_channel_input, n_channel_output, n_filters):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(n_channel_input, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_filters*4, n_filters*8, 4, 2, 1)
        self.conv5 = nn.Conv2d(n_filters*8, n_filters*16, 4, 2, 1)
        self.conv6 = nn.Conv2d(n_filters*16,n_filters*16,4,2,1)

        self.deconv1 = nn.ConvTranspose2d(n_filters*16,n_filters*16,4,2,1)
        self.deconv2 = nn.ConvTranspose2d(n_filters*8*4, n_filters*8, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(n_filters*8*2, n_filters*4, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(n_filters*8, n_filters*2, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(n_filters*4, n_filters, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(n_filters*2, n_channel_output, 4, 2, 1)

        self.norm1 = nn.BatchNorm2d(n_filters)
        self.norm2 = nn.BatchNorm2d(n_filters*2)
        self.norm3 = nn.BatchNorm2d(n_filters*4)
        self.norm4 = nn.BatchNorm2d(n_filters*8)
        self.norm5 = nn.BatchNorm2d(n_filters*16)

        self.leakyrelu = nn.LeakyReLU(0.1,True)
        self.relu = nn.ReLU(True)
        
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
    
    def forward(self, input):
        encoder1 = self.conv1(input)
        encoder2 = self.norm2(self.conv2(self.leakyrelu(encoder1)))
        encoder3 = self.norm3(self.conv3(self.leakyrelu(encoder2)))
        encoder4 = self.norm4(self.conv4(self.leakyrelu(encoder3)))
        encoder5 = self.norm5(self.conv5(self.leakyrelu(encoder4)))
        encoder6 = self.conv6(self.leakyrelu(encoder5))


        decoder1 = self.dropout(self.norm5(self.deconv1(self.relu(encoder6))))
        decoder1 = torch.cat((decoder1,encoder5),1)
        decoder2 = self.dropout(self.norm4(self.deconv2(self.relu(decoder1))))
        decoder2 = torch.cat((decoder2,encoder4),1)
        decoder3 = self.dropout(self.norm3(self.deconv3(self.relu(decoder2))))
        decoder3 = torch.cat((decoder3,encoder3),1)
        decoder4 = self.dropout(self.norm2(self.deconv4(self.relu(decoder3))))
        decoder4 = torch.cat((decoder4,encoder2),1)
        decoder5 = self.dropout(self.norm1(self.deconv5(self.relu(decoder4))))
        decoder5 = torch.cat((decoder5,encoder1),1)
        decoder6 = self.deconv6(self.relu(decoder5))
        output = self.tanh(decoder6)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, n_channel_input, n_channel_output, n_filters):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(n_channel_input + n_channel_output, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_filters * 4, n_filters * 8, 4, 1, 1)
        self.conv5 = nn.Conv2d(n_filters * 8, 1, 4, 1, 1)

        self.batch_norm2 = nn.BatchNorm2d(n_filters * 2)
        self.batch_norm4 = nn.BatchNorm2d(n_filters * 4)
        self.batch_norm8 = nn.BatchNorm2d(n_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.1, True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        encoder1 = self.conv1(input)
        encoder2 = self.batch_norm2(self.conv2(self.leaky_relu(encoder1)))
        encoder3 = self.batch_norm4(self.conv3(self.leaky_relu(encoder2)))
        encoder4 = self.batch_norm8(self.conv4(self.leaky_relu(encoder3)))
        encoder5 = self.conv5(self.leaky_relu(encoder4))
        output =  self.sigmoid(encoder5)
        return output
