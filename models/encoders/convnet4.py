from collections import OrderedDict

import torch.nn as nn

from .encoders import register
from ..modules import *


__all__ = ['convnet4', 'wide_convnet4']


class ConvBlock(Module):
  def __init__(self, in_channels, out_channels, bn_args):
    super(ConvBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.conv = Conv2d(in_channels, out_channels, 3, 1, padding=1)
    self.bn = BatchNorm2d(out_channels, **bn_args)
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(2)

  def forward(self, x, params=None, episode=None):
    out = self.conv(x, get_child_dict(params, 'conv'))
    out = self.bn(out, get_child_dict(params, 'bn'), episode)
    out = self.pool(self.relu(out))
    return out


class ConvNet4(Module):
  def __init__(self, hid_dim, out_dim, bn_args):
    super(ConvNet4, self).__init__()
    self.hid_dim = hid_dim
    self.out_dim = out_dim

    episodic = bn_args.get('episodic') or []
    bn_args_ep, bn_args_no_ep = bn_args.copy(), bn_args.copy()
    bn_args_ep['episodic'] = True
    bn_args_no_ep['episodic'] = False
    bn_args_dict = dict()
    for i in [1, 2, 3, 4]:
      if 'conv%d' % i in episodic:
        bn_args_dict[i] = bn_args_ep
      else:
        bn_args_dict[i] = bn_args_no_ep

    self.encoder = Sequential(OrderedDict([
      ('conv1', ConvBlock(3, hid_dim, bn_args_dict[1])),
      ('conv2', ConvBlock(hid_dim, hid_dim, bn_args_dict[2])),
      ('conv3', ConvBlock(hid_dim, hid_dim, bn_args_dict[3])),
      ('conv4', ConvBlock(hid_dim, out_dim, bn_args_dict[4])),
    ]))

  def get_out_dim(self, scale=25):
    return self.out_dim * scale

  def forward(self, x, params=None, episode=None):
    out = self.encoder(x, get_child_dict(params, 'encoder'), episode)
    out = out.view(out.shape[0], -1)
    return out


@register('convnet4')
def convnet4(bn_args=dict()):
  return ConvNet4(32, 32, bn_args)


@register('wide-convnet4')
def wide_convnet4(bn_args=dict()):
  return ConvNet4(64, 64, bn_args)