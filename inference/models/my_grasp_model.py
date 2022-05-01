import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.pooling import AvgPool2d
from torchsummary import summary
import math

class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }


class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class OSAModule(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(OSAModule, self).__init__()
        self.droprate = dropRate

        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return out
        
class OSABlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(OSABlock, self).__init__()
        self.nb_layers = nb_layers
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes, growth_rate, dropRate))
            in_planes = growth_rate
            if i == (nb_layers-1):
                layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out_put = []
        out = self.layer[0](x)
        out_put.append(out)

        for i in range(1, self.nb_layers):
            out = self.layer[i](out)
            out_put.append(out)
            if i == (self.nb_layers):
                out = self.layer[i](torch.cat([out_put], 1))
        return out

class OSADenseNet(nn.Module):
    def __init__(self, depth, growth_rate=32,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(OSADenseNet, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = OSAModule
        else:
            block = BasicBlock
        n = int(n)
        print("n ", n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(4, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = OSABlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = OSABlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = OSABlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        # out = out.view(-1, self.in_planes)
        # return self.fc(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        print("block ", block)

        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, growth_rate=32,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        print("n ", n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(4, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        # out = out.view(-1, self.in_planes)
        # return self.fc(out)
        return out

# if __name__ == "__main__":
    # net = DenseNet3(50, 12, reduction=0.5).to("cuda")
    
    # net = OSADenseNet(depth=50, growth_rate=12, reduction=0.5).to("cuda")
    # net = OSABlock(5, 4, 32, OSAModule).to("cuda")
    # net = OSAModule(4, 32).to("cuda")

    # summary(net, (4, 224, 224))     
    # print(net.state_dict()['bn1.weight'].shape[0])   
    # while 1:
    #     pass



# class BN_Conv2d(nn.Module):
#     def __init__(self, in_channels:object, out_channels:object, kernel_size:object, 
#                 stride:object, padding:object, dilation=1, groups=1, bias=False) -> object:
        
#         super(BN_Conv2d, self).__init__()

#         self.seq = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                         padding=padding, dilation=dilation, groups=groups, bias=bias),
#             nn.BatchNorm2d(out_channels))

#     def forward(self, x):
#         return F.relu(self.seq(x))

# class DenseBlock(nn.Module):
#     def __init__(self, input_channels, num_layers, growth_rate):
        
#         super(DenseBlock, self).__init__()

#         self.num_layers = num_layers
#         self.k0 = input_channels
#         self.k = growth_rate
#         self.layers = self.__make__layers()
    
#     def __make__layers(self):
#         layer_list = nn.ModuleList()
#         for i in range(self.num_layers):
#             layer_list.append(nn.Sequential(
#                 BN_Conv2d(self.k0+i*self.k, 4*self.k, 1, 1, 0),
#                 BN_Conv2d(4*self.k, self.k, 3, 1, 1)
#             ))
#         return layer_list
    
#     def forward(self, x):
#         feature = self.layers[0](x)
#         out = torch.cat((x, feature), 1)
#         for i in range(1, len(self.layers)):
#             feature = self.layers[i](out)
#             out = torch.cat((feature, out), 1)
#         return out

# class DenseNet(nn.Module):
#     def __init__(self, layers:object, k, theta, num_classes)->object:
#         super(DenseNet, self).__init__()

#         #params
#         self.layers = layers
#         self.k = k
#         self.theta = theta
        
#         #layers
#         self.conv = BN_Conv2d(3, 2*k, 7, 2, 3)
#         self.blocks, patches = self.__make_blocks(2*k)
#         self.fc = nn.Linear(patches, num_classes)

#     def __make_transition(self, in_chls):
#         out_chls = int(self.theta*in_chls)
#         return nn.Sequential(
#             BN_Conv2d(in_chls, out_chls, 1, 1, 0),
#             nn.AvgPool2d(2)
#         ), out_chls

#     def __make_blocks(self, k0):
#         layers_list = nn.ModuleList()
#         patches = 0
#         for i in range(len(self.layers)):
#             layers_list.append(DenseBlock(k0, self.layers[i], self.k))
#             #output feature patches from Dnse Block
#             patches = k0 + self.layers[i]*self.k
#             if i != len(self.layers)-1:
#                 transition, k0 = self.__make_transition(patches)
#                 layers_list.append(transition)
#         return nn.Sequential(*layers_list), patches
    
#     def forward(self, x):
#         out = self.conv(x)
#         out = F.max_pool2d(out, 3, 2 ,1)
#         # print(out.shape)
#         out = self.blocks(out)
#         # print(out.shape)
#         out = F.avg_pool2d(out, 7)
#         # print(out.shape)
#         out = out.view(out.size(0), -1)
#         out = F.softmax(self.fc(out))

#         return out

# class GraspDenseNet(nn.Module):
#     def __init__(self, layers:object, input_channels, channel_size, k, theta)->object:
#         super(GraspDenseNet, self).__init__()

#         #params
#         self.layers = layers
#         self.k = k
#         self.theta = theta
        
#         #layers
#         self.conv = BN_Conv2d(input_channels, input_channels, channel_size, 1, 4)
#         self.blocks, patches = self.__make_blocks(input_channels)

#     def __make_transition(self, in_chls):
#         out_chls = int(self.theta*in_chls)
#         return nn.Sequential(
#             BN_Conv2d(in_chls, out_chls, 1, 1, 0),
#             nn.AvgPool2d(2)
#         ), out_chls

#     def __make_blocks(self, k0):
#         layers_list = nn.ModuleList()
#         patches = 0

#         for i in range(len(self.layers)):
#             layers_list.append(DenseBlock(k0, self.layers[i], self.k))
#             #output feature patches from Dnse Block
#             patches = k0 + self.layers[i]*self.k
#             if i != len(self.layers)-1:
#                 transition, k0 = self.__make_transition(patches)
#                 layers_list.append(transition)
#         return nn.Sequential(*layers_list), patches
    
#     def forward(self, x):
#         # out = self.conv(x)
#         # out = F.max_pool2d(out, 3, 2 ,1)
#         # print(out.shape)
#         out = self.blocks(x)
#         # print(out.shape)
#         # out = F.avg_pool2d(out, 7)
#         # print(out.shape)
#         # out = out.view(out.size(0), -1)

#         return out







# class Dense_Block(nn.Module):
#     def __init__(self, in_channels):
#         super(Dense_Block, self).__init__()
#         self.relu = nn.ReLU(inplace = True)
#         self.bn = nn.BatchNorm2d(in_channels)

#         self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
#         self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
#         self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
#         self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
#         self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

#     def forward(self, x):
#         bn = self.bn(x) 
#         conv1 = self.relu(self.conv1(bn))
#         conv2 = self.relu(self.conv2(conv1))
#         # Concatenate in channel dimension
#         c2_dense = self.relu(torch.cat([conv1, conv2], 1))
#         conv3 = self.relu(self.conv3(c2_dense))
#         c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

#         conv4 = self.relu(self.conv4(c3_dense)) 
#         c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

#         conv5 = self.relu(self.conv5(c4_dense))
#         c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

#         return c5_dense

# class Transition_Layer(nn.Module): 
#     def __init__(self, in_channels, out_channels):
#         super(Transition_Layer, self).__init__() 

#         self.relu = nn.ReLU(inplace = True) 
#         self.bn = nn.BatchNorm2d(out_channels) 
#         self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False) 
#         self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0) 

#     def forward(self, x): 
#         bn = self.bn(self.relu(self.conv(x))) 
#         out = self.avg_pool(bn) 
#         return out 

# class DenseNet(nn.Module): 
#     def __init__(self, nr_classes): 
#         super(DenseNet, self).__init__() 

#         self.lowconv = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 7, padding = 3, bias = False) 
#         self.relu = nn.ReLU()

#         # Make Dense Blocks 
#         self.denseblock1 = self._make_dense_block(Dense_Block, 64) 
#         self.denseblock2 = self._make_dense_block(Dense_Block, 128)
#         self.denseblock3 = self._make_dense_block(Dense_Block, 128)
#         # Make transition Layers 
#         self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128) 
#         self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128) 
#         self.transitionLayer3 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 64)
#         # Classifier 
#         self.bn = nn.BatchNorm2d(num_features = 64) 
#         self.pre_classifier = nn.Linear(64*4*4, 512) 
#         self.classifier = nn.Linear(512, nr_classes)

#     def _make_dense_block(self, block, in_channels): 
#         layers = [] 
#         layers.append(block(in_channels)) 
#         return nn.Sequential(*layers) 

#     def _make_transition_layer(self, layer, in_channels, out_channels): 
#         modules = [] 
#         modules.append(layer(in_channels, out_channels)) 
#         return nn.Sequential(*modules) 

#     def forward(self, x): 
#         out = self.relu(self.lowconv(x)) 
#         out = self.denseblock1(out) 
#         out = self.transitionLayer1(out) 
#         out = self.denseblock2(out) 
#         out = self.transitionLayer2(out) 

#         out = self.denseblock3(out) 
#         # out = self.transitionLayer3(out) 

#         # out = self.bn(out) 
#         # out = out.view(-1, 64*4*4) 

#         # out = self.pre_classifier(out) 
#         # out = self.classifier(out)
#         return out