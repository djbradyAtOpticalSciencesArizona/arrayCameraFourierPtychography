import torch.nn as nn
import torch

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.PReLU()),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.PReLU()),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.PReLU())
        self.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        
        self.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2,padding=1))
#         self.add_module('norm2', nn.BatchNorm2d(num_output_features))
#         self.add_module('relu2', nn.LeakyReLU(0.1))
#         self.add_module('conv2', nn.Conv2d(num_output_features, num_output_features,
#                                           kernel_size=3, stride=2, padding=1,bias=False))
        
class _Transition_up(nn.Sequential):
    def __init__(self, num_input_features, num_output_features,out_padding):
        super(_Transition_up, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.PReLU())
        self.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(num_output_features))
        self.add_module('relu2', nn.PReLU())
        self.add_module('conv2', nn.ConvTranspose2d(num_output_features, num_output_features,
                                                    kernel_size=3,stride=2,padding=1,output_padding=out_padding,bias=False))
        
class ReconNet(nn.Module):
    def __init__(self, num_input_feature = 16, num_init_features=64,growth_rate=24,
                 bn_size=4,depth_of_block=5, compression_downsample=0.8,compression_upsample=0.2):

        super(ReconNet, self).__init__()
        assert 0 < compression_downsample <= 1, 'compression of densenet should be between 0 and 1'
        assert 0 < compression_upsample <= 1, 'compression of densenet should be between 0 and 1'
        
        self.conv0 = nn.Conv2d(num_input_feature,num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        
        num_channel1 = num_init_features
        self.dense_down1 = _DenseBlock(depth_of_block,num_channel1,bn_size=bn_size,growth_rate=growth_rate)
        self.downsample1 = _Transition(num_channel1 + depth_of_block * growth_rate,
                                       int((num_channel1 + depth_of_block * growth_rate) * compression_downsample))
        
        num_channel2 = int((num_channel1 + depth_of_block * growth_rate) * compression_downsample)
        self.dense_down2 = _DenseBlock(depth_of_block,num_channel2,bn_size=bn_size,growth_rate=growth_rate)
        self.downsample2 = _Transition(num_channel2 + depth_of_block * growth_rate,
                                       int((num_channel2 + depth_of_block * growth_rate) * compression_downsample))

        num_channel3 = int((num_channel2 + depth_of_block * growth_rate) * compression_downsample)
        self.dense_down3 = _DenseBlock(depth_of_block,num_channel3,bn_size=bn_size,growth_rate=growth_rate)
        self.downsample3 = _Transition(num_channel3 + depth_of_block * growth_rate,
                                       int((num_channel3 + depth_of_block * growth_rate) * compression_downsample))

        num_channel4 = int((num_channel3 + depth_of_block * growth_rate) * compression_downsample)
        self.dense_down4 = _DenseBlock(depth_of_block,num_channel4,bn_size=bn_size,growth_rate=growth_rate)
        self.downsample4 = _Transition(num_channel4 + depth_of_block * growth_rate,
                                       int((num_channel4 + depth_of_block * growth_rate) * compression_downsample))

        num_channel5 = int((num_channel4 + depth_of_block * growth_rate) * compression_downsample)
        self.dense_down5 = _DenseBlock(depth_of_block,num_channel5,bn_size=bn_size,growth_rate=growth_rate)
        self.downsample5 = _Transition(num_channel5 + depth_of_block * growth_rate,
                                       int((num_channel5 + depth_of_block * growth_rate) * compression_downsample))

        num_channel6 = int((num_channel5 + depth_of_block * growth_rate) * compression_downsample)
        self.dense_down6 = _DenseBlock(depth_of_block,num_channel6,bn_size=bn_size,growth_rate=growth_rate)
        self.downsample6 = _Transition(num_channel6 + depth_of_block * growth_rate,
                                       int((num_channel6 + depth_of_block * growth_rate) * compression_downsample))

        num_channel7 = int((num_channel6 + depth_of_block * growth_rate) * compression_downsample)
        self.dense_down7 = _DenseBlock(depth_of_block,num_channel7,bn_size=bn_size,growth_rate=growth_rate)
        self.downsample7 = _Transition(num_channel7 + depth_of_block * growth_rate,
                                       int((num_channel7 + depth_of_block * growth_rate) * compression_downsample))

        num_channel8 = int((num_channel7 + depth_of_block * growth_rate) * compression_downsample)
        self.dense_up1 = _DenseBlock(depth_of_block,num_channel8,bn_size=bn_size,growth_rate=growth_rate)
        self.upsample1 = _Transition_up(num_channel8 + depth_of_block * growth_rate,
                                        int((num_channel8 + depth_of_block * growth_rate) * compression_upsample),out_padding=(0,1))

        num_channel9 = int((num_channel8 + depth_of_block * growth_rate) * compression_upsample) + num_channel7 + depth_of_block * growth_rate
        self.dense_up2 = _DenseBlock(depth_of_block,num_channel9,bn_size=bn_size,growth_rate=growth_rate)
        self.upsample2 = _Transition_up(num_channel9 + depth_of_block * growth_rate,
                                        int((num_channel9 + depth_of_block * growth_rate) * compression_upsample),out_padding=1)

        num_channel10 = int((num_channel9 + depth_of_block * growth_rate) * compression_upsample) + num_channel6 + depth_of_block * growth_rate
        self.dense_up3 = _DenseBlock(depth_of_block,num_channel10,bn_size=bn_size,growth_rate=growth_rate)
        self.upsample3 = _Transition_up(num_channel10 + depth_of_block * growth_rate,
                                        int((num_channel10 + depth_of_block * growth_rate) * compression_upsample),out_padding=1)

        num_channel11 = int((num_channel10 + depth_of_block * growth_rate) * compression_upsample) + num_channel5 + depth_of_block * growth_rate
        self.dense_up4 = _DenseBlock(depth_of_block,num_channel11,bn_size=bn_size,growth_rate=growth_rate)
        self.upsample4 = _Transition_up(num_channel11 + depth_of_block * growth_rate,
                                        int((num_channel11 + depth_of_block * growth_rate) * compression_upsample),out_padding=1)

        num_channel12 = int((num_channel11 + depth_of_block * growth_rate) * compression_upsample) + num_channel4 + depth_of_block * growth_rate
        self.dense_up5 = _DenseBlock(depth_of_block,num_channel12,bn_size=bn_size,growth_rate=growth_rate)
        self.upsample5 = _Transition_up(num_channel12 + depth_of_block * growth_rate,
                                        int((num_channel12 + depth_of_block * growth_rate) * compression_upsample),out_padding=1)

        num_channel13 = int((num_channel12 + depth_of_block * growth_rate) * compression_upsample) + num_channel3 + depth_of_block * growth_rate
        self.dense_up6 = _DenseBlock(depth_of_block,num_channel13,bn_size=bn_size,growth_rate=growth_rate)
        self.upsample6 = _Transition_up(num_channel13 + depth_of_block * growth_rate,
                                        int((num_channel13 + depth_of_block * growth_rate) * compression_upsample),out_padding=1)

        num_channel14 = int((num_channel13 + depth_of_block * growth_rate) * compression_upsample) + num_channel2 + depth_of_block * growth_rate
        self.dense_up7 = _DenseBlock(depth_of_block,num_channel14,bn_size=bn_size,growth_rate=growth_rate)
        self.upsample7 = _Transition_up(num_channel14 + depth_of_block * growth_rate,
                                        int((num_channel14 + depth_of_block * growth_rate) * compression_upsample),out_padding=1)
        
        num_channel15 = int((num_channel14 + depth_of_block * growth_rate) * compression_upsample) + num_channel1 + depth_of_block * growth_rate
        self.final = nn.Sequential(_DenseBlock(depth_of_block,num_channel15,bn_size=bn_size,growth_rate=growth_rate),
                                   nn.BatchNorm2d(num_channel15 + depth_of_block * growth_rate),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv2d(num_channel15 + depth_of_block * growth_rate,1,kernel_size=1, stride=1),
                                   nn.Sigmoid())
    def forward(self, x):
        x1 = self.dense_down1(self.conv0(x))
        x2 = self.dense_down2(self.downsample1(x1))
        x3 = self.dense_down3(self.downsample2(x2))
        x4 = self.dense_down4(self.downsample3(x3))
        x5 = self.dense_down5(self.downsample4(x4))
        x6 = self.dense_down6(self.downsample5(x5))
        x7 = self.dense_down7(self.downsample6(x6))
        x8 = self.dense_up1(self.downsample7(x7))
        x9 = torch.cat([x7,self.upsample1(x8)],1)
        x9 = self.dense_up2(x9)
        x10 = torch.cat([x6,self.upsample2(x9)],1)
        x10 = self.dense_up3(x10)
        x11 = torch.cat([x5,self.upsample3(x10)],1)
        x11 = self.dense_up4(x11)
        x12 = torch.cat([x4,self.upsample4(x11)],1)
        x12 = self.dense_up5(x12)
        x13 = torch.cat([x3,self.upsample5(x12)],1)
        x13 = self.dense_up6(x13)
        x14 = torch.cat([x2,self.upsample6(x13)],1)
        x14 = self.dense_up7(x14)
        x15 = torch.cat([x1,self.upsample7(x14)],1)
        x = self.final(x15)
        return x