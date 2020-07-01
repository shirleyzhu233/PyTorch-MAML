import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Module', 'Conv2d', 'Linear', 'BatchNorm2d', 'Sequential', 
           'get_child_dict']


def get_child_dict(params, key=None):
  """
  Constructs parameter dictionary for a network module.

  Args:
    params (dict): a parent dictionary of named parameters.
    key (str, optional): a key that specifies the root of the child dictionary.

  Returns:
    child_dict (dict): a child dictionary of model parameters.
  """
  if params is None:
    return None
  if key is None or (isinstance(key, str) and key == ''):
    return params

  key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
  if not any(filter(key_re.match, params.keys())):  # handles nn.DataParallel
    key_re = re.compile(r'^module\.{0}\.(.+)'.format(re.escape(key)))
  child_dict = OrderedDict(
    (key_re.sub(r'\1', k), value) for (k, value)
      in params.items() if key_re.match(k) is not None)
  return child_dict


class Module(nn.Module):
  def __init__(self):
    super(Module, self).__init__()
    self.efficient = False
    self.first_pass = True

  def go_efficient(self, mode=True):
    """ Switches on / off gradient checkpointing. """
    self.efficient = mode
    for m in self.children():
      if isinstance(m, Module):
        m.go_efficient(mode)

  def is_first_pass(self, mode=True):
    """ Tracks the progress of forward and backward pass when gradient 
    checkpointing is enabled. """
    self.first_pass = mode
    for m in self.children():
      if isinstance(m, Module):
        m.is_first_pass(mode)


class Conv2d(nn.Conv2d, Module):
  def __init__(self, in_channels, out_channels, kernel_size, 
               stride=1, padding=0, bias=True):
    super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, 
                                 stride, padding, bias=bias)

  def forward(self, x, params=None, episode=None):
    if params is None:
      x = super(Conv2d, self).forward(x)
    else:
      weight, bias = params.get('weight'), params.get('bias')
      if weight is None:
        weight = self.weight
      if bias is None:
        bias = self.bias
      x = F.conv2d(x, weight, bias, self.stride, self.padding)
    return x


class Linear(nn.Linear, Module):
  def __init__(self, in_features, out_features, bias=True):
    super(Linear, self).__init__(in_features, out_features, bias=bias)

  def forward(self, x, params=None, episode=None):
    if params is None:
      x = super(Linear, self).forward(x)
    else:
      weight, bias = params.get('weight'), params.get('bias')
      if weight is None:
        weight = self.weight
      if bias is None:
        bias = self.bias
      x = F.linear(x, weight, bias)
    return x


class BatchNorm2d(nn.BatchNorm2d, Module):
  def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, 
               track_running_stats=True, episodic=False, n_episode=4,
               alpha=False):
    """
    Args:
      episodic (bool, optional): if True, maintains running statistics for 
        each episode separately. It is ignored if track_running_stats=False. 
        Default: True
      n_episode (int, optional): number of episodes per mini-batch. It is 
        ignored if episodic=False.
      alpha (bool, optional): if True, learns to interpolate between batch 
        statistics computed over the support set and instance statistics from 
        a query at validation time. Default: True
        (It is ignored if track_running_stats=False or meta_learn=False)
    """
    super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, 
                                      track_running_stats)
    self.episodic = episodic
    self.n_episode = n_episode
    self.alpha = alpha

    if self.track_running_stats:
      if self.episodic:
        for ep in range(n_episode):
          self.register_buffer(
            'running_mean_%d' % ep, torch.zeros(num_features))
          self.register_buffer(
            'running_var_%d' % ep, torch.ones(num_features))
          self.register_buffer(
            'num_batches_tracked_%d' % ep, torch.tensor(0, dtype=torch.int))
      if self.alpha:
        self.register_buffer('batch_size', torch.tensor(0, dtype=torch.int))
        self.alpha_scale = nn.Parameter(torch.tensor(0.))
        self.alpha_offset = nn.Parameter(torch.tensor(0.))
        
  def is_episodic(self):
    return self.episodic

  def _batch_norm(self, x, mean, var, weight=None, bias=None):
    if self.affine:
      assert weight is not None and bias is not None
      weight = weight.view(1, -1, 1, 1)
      bias = bias.view(1, -1, 1, 1)
      x = weight * (x - mean) / (var + self.eps) ** .5 + bias
    else:
      x = (x - mean) / (var + self.eps) ** .5
    return x

  def reset_episodic_running_stats(self, episode):
    if self.episodic:
      getattr(self, 'running_mean_%d' % episode).zero_()
      getattr(self, 'running_var_%d' % episode).fill_(1.)
      getattr(self, 'num_batches_tracked_%d' % episode).zero_()

  def forward(self, x, params=None, episode=None):
    self._check_input_dim(x)
    if params is not None:
      weight, bias = params.get('weight'), params.get('bias')
      if weight is None:
        weight = self.weight
      if bias is None:
        bias = self.bias
    else:
      weight, bias = self.weight, self.bias

    if self.track_running_stats:
      if self.episodic:
        assert episode is not None and episode < self.n_episode
        running_mean = getattr(self, 'running_mean_%d' % episode)
        running_var = getattr(self, 'running_var_%d' % episode)
        num_batches_tracked = getattr(self, 'num_batches_tracked_%d' % episode)
      else:
        running_mean, running_var = self.running_mean, self.running_var
        num_batches_tracked = self.num_batches_tracked

      if self.training:
        exp_avg_factor = 0.
        if self.first_pass: # only updates statistics in the first pass
          if self.alpha:
            self.batch_size = x.size(0)
          num_batches_tracked += 1
          if self.momentum is None:
            exp_avg_factor = 1. / float(num_batches_tracked)
          else:
            exp_avg_factor = self.momentum
        return F.batch_norm(x, running_mean, running_var, weight, bias,
                            True, exp_avg_factor, self.eps)
      else:
        if self.alpha:
          assert self.batch_size > 0
          alpha = torch.sigmoid(
            self.alpha_scale * self.batch_size + self.alpha_offset)
          # exponentially moving-averaged training statistics
          running_mean = running_mean.view(1, -1, 1, 1)
          running_var = running_var.view(1, -1, 1, 1)
          # per-sample statistics
          sample_mean = torch.mean(x, dim=(2, 3), keepdim=True)
          sample_var = torch.var(x, dim=(2, 3), unbiased=False, keepdim=True)
          # interpolated statistics
          mean = alpha * running_mean + (1 - alpha) * sample_mean
          var = alpha * running_var + (1 - alpha) * sample_var + \
                alpha * (1 - alpha) * (sample_mean - running_mean) ** 2
          return self._batch_norm(x, mean, var, weight, bias)
        else:
          return F.batch_norm(x, running_mean, running_var, weight, bias,
                              False, 0., self.eps)
    else:
      return F.batch_norm(x, None, None, weight, bias, True, 0., self.eps)


class Sequential(nn.Sequential, Module):
  def __init__(self, *args):
    super(Sequential, self).__init__(*args)

  def forward(self, x, params=None, episode=None):
    if params is None:
      for module in self:
        x = module(x, None, episode)
    else:
      for name, module in self._modules.items():
        x = module(x, get_child_dict(params, name), episode)
    return x