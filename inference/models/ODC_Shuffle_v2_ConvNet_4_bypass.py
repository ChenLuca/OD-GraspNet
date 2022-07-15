import torch.nn as nn
import torch.nn.functional as F
import torch
from inference.models.RJ_grasp_model import GraspModel, OSAModule, OSABlock, TransitionBlock
from inference.models.cbam import CBAM

class Generative_ODC_Shuffle_v2_4_bypass(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=256, dropout=False, prob=0.0):
        super(Generative_ODC_Shuffle_v2_4_bypass, self).__init__()

        print("Generative_ODC_Shuffle_v2_4_bypass")

        self.shuffle_factor = [2, 2]
        self.osa_depth = 5
        self.osa_conv_kernal = [64, 40, 48, 56]
        self.trans_conv_kernal = [128, 128, 192, 256]
        self.osa_drop_rate = 0.0
        self.osa_reduction = 1.0

        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size*2, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.bn2 = nn.BatchNorm2d(channel_size*2)

        self.conv3 = nn.Conv2d(channel_size*2, channel_size * 4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)


         # 1st block
        self.block1 = OSABlock(self.osa_depth,channel_size * 4, self.osa_conv_kernal[0], OSAModule, self.osa_drop_rate)
        self.trans1 = TransitionBlock(self.osa_conv_kernal[0]*self.osa_depth, self.trans_conv_kernal[0], dropRate=self.osa_drop_rate)
        self.cbam1 = CBAM(self.trans_conv_kernal[0])


        self.conv4 = nn.Conv2d(((channel_size * 4) +(channel_size * 2)), self.trans_conv_kernal[0], kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(self.trans_conv_kernal[0])
        self.cbam2 = CBAM(self.trans_conv_kernal[0])

        self.conv5 = nn.Conv2d(self.trans_conv_kernal[0], channel_size*4, kernel_size=3, stride=1, padding=1)
        self.sf1 = nn.PixelShuffle(self.shuffle_factor[1])
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.Conv2d((channel_size + channel_size), channel_size*2, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(channel_size*2)
        self.cbam3 = CBAM(channel_size*2)

        self.conv7 = nn.Conv2d(channel_size*2, channel_size*2, kernel_size=3, stride=1, padding=1)
        self.sf2 = nn.PixelShuffle(self.shuffle_factor[0])
        self.bn7 = nn.BatchNorm2d(16)

        self.pos_output = nn.Conv2d(in_channels=16, out_channels=output_channels, kernel_size=3, padding=1)
        self.cos_output = nn.Conv2d(in_channels=16, out_channels=output_channels, kernel_size=3, padding=1)
        self.sin_output = nn.Conv2d(in_channels=16, out_channels=output_channels, kernel_size=3, padding=1)
        self.width_output = nn.Conv2d(in_channels=16, out_channels=output_channels, kernel_size=3, padding=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)


    def forward(self, x_in):

        x_1 = F.relu(self.bn1(self.maxpool1(self.conv1(x_in))))
        x_2 = F.relu(self.bn2(self.maxpool2(self.conv2(x_1))))
        x_3 = F.relu(self.bn3(self.conv3(x_2)))

        x = self.trans1(self.block1(x_3))
        x = self.cbam1(x)

        x = torch.cat((x, x_2), 1)
        x = F.relu((self.bn4(self.conv4(x))))
        x = self.cbam2(x)

        x = F.relu(self.bn5(self.sf1(self.conv5(x))))

        x = torch.cat((x, x_1), 1)
        x = F.relu((self.bn6(self.conv6(x))))
        x = self.cbam3(x)

        x = F.relu(self.bn7(self.sf2(self.conv7(x))))

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
