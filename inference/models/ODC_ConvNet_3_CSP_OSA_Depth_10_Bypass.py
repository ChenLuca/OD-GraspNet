import torch.nn as nn
import torch.nn.functional as F
import torch
from inference.models.RJ_grasp_model import GraspModel, OSAModule, OSABlock, TransitionBlock
from inference.models.cbam import CBAM

class Generative_ODC_3_CSP_OSA_Depth_10_Bypass(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=256, dropout=False, prob=0.0):
        super(Generative_ODC_3_CSP_OSA_Depth_10_Bypass, self).__init__()

        print("Generative_ODC_3_CSP_OSA_Depth_10_Bypass")
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.conv_out_size = channel_size * 4

        #osa-dense
        self.osa_depth = 10
        self.osa_conv_kernal = [32, 40, 48, 56]
        self.trans_conv_kernal = [64, 128, 192, 256]
        self.osa_drop_rate = 0.0
        self.osa_reduction = 1.0
        
        #csp
        self.csp_part_ratio = 0.5

        self.part_1_channels_1 = int(self.conv_out_size*self.csp_part_ratio)
        self.part_2_channels_1 = self.conv_out_size - self.part_1_channels_1

        self.csp_trans1_out_kernal = self.trans_conv_kernal[0] * 2 

        self.part_1_channels_2 = int(self.csp_trans1_out_kernal*self.csp_part_ratio)
        self.part_2_channels_2 = self.csp_trans1_out_kernal - self.part_1_channels_2

        self.csp_trans2_out_kernal = self.trans_conv_kernal[1] * 2

        self.concat_1_size = self.conv_out_size + self.csp_trans2_out_kernal
        self.concat_2_size = (channel_size * 2) +  (channel_size * 2)

        # 1st block
        self.block1 = OSABlock(self.osa_depth, self.part_2_channels_1, self.osa_conv_kernal[0], OSAModule, self.osa_drop_rate)
        self.osa_trans1 = TransitionBlock(self.osa_conv_kernal[0]*self.osa_depth, self.trans_conv_kernal[0], dropRate=self.osa_drop_rate)
        self.csp_trans1 = TransitionBlock(self.part_1_channels_1, self.trans_conv_kernal[0], dropRate=self.osa_drop_rate)

        self.cbam1 = CBAM(self.csp_trans1_out_kernal)
        
        # 2nd block
        self.block2 = OSABlock(self.osa_depth, self.part_2_channels_2, self.osa_conv_kernal[1], OSAModule, self.osa_drop_rate)
        self.osa_trans2 = TransitionBlock(self.osa_conv_kernal[1]*self.osa_depth, self.trans_conv_kernal[1], dropRate=self.osa_drop_rate)
        self.csp_trans2 = TransitionBlock(self.part_1_channels_2, self.trans_conv_kernal[1], dropRate=self.osa_drop_rate)

        self.cbam2 = CBAM(self.csp_trans2_out_kernal)

        self.conv4 = nn.ConvTranspose2d(self.concat_1_size, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=0)

        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(self.concat_2_size, channel_size, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)

        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):

        x_1 = F.relu(self.bn1(self.conv1(x_in)))
        x_2 = F.relu(self.bn2(self.conv2(x_1)))
        x_3 = F.relu(self.bn3(self.conv3(x_2)))

        part_1 = x_3[:,:self.part_1_channels_1, :, :]
        part_2 = x_3[:,self.part_1_channels_1:, :, :]
        x_part_1 = self.csp_trans1(part_1)
        x_part_2 = self.osa_trans1(self.block1(part_2))
        x = torch.cat((x_part_1, x_part_2), 1)
        # print("part_1.shape ", part_1.shape)
        # print("part_2.shape ", part_2.shape)
        # print("cat_1.shape ", x.shape)
        x = self.cbam1(x)

        part_1 = x[:,:self.part_1_channels_2, :, :]
        part_2 = x[:,self.part_1_channels_2:, :, :]
        x_part_1 = self.csp_trans2(part_1)
        x_part_2 = self.osa_trans2(self.block2(part_2))
        x = torch.cat((x_part_1, x_part_2), 1)
        # print("part_1.shape ", part_1.shape)
        # print("part_2.shape ", part_2.shape)
        # print("cat_2.shape ", x.shape)
        x = self.cbam2(x)

        x = torch.cat((x, x_3), 1)
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.cat((x, x_2), 1)
        x = F.relu(self.bn5(self.conv5(x)))

        x = self.conv6(x)

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
