

import torch.nn as nn


def conv_bn_relu(in_planes, out_planes, kernel=3, stride=1, padding=1):
     net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
                         nn.BatchNorm2d(num_features=out_planes),
                         nn.ReLU(True))
     return net;

# input size Bx3x224x224
class SegmenterModel(nn.Module):
    def __init__(self, in_size=3):
        super(SegmenterModel, self).__init__()
        
        # implement your model here
         
    def forward(self, input):
        return input

