import os
import shutil
import time

import numpy as np
import scipy.stats as stats


_log_path = None

def set_log_path(path):
  global _log_path
  _log_path = path


def log(obj, filename='log.txt'):
  print(obj)
  if _log_path is not None:
    with open(os.path.join(_log_path, filename), 'a') as f:
      print(obj, file=f)


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.
    self.avg = 0.
    self.sum = 0.
    self.count = 0.

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def item(self):
    return self.avg


class Timer(object):
  def __init__(self):
    self.start()

  def start(self):
    self.v = time.time()

  def end(self):
    return time.time() - self.v


def set_gpu(gpu):
  print('set gpu:', gpu)
  os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=True):
  basename = os.path.basename(path.rstrip('/'))
  if os.path.exists(path):
    if remove and (basename.startswith('_')
      or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
      shutil.rmtree(path)
      os.makedirs(path)
  else:
    os.makedirs(path)


def time_str(t):
  if t >= 3600:
    return '{:.1f}h'.format(t / 3600)
  if t >= 60:
    return '{:.1f}m'.format(t / 60)
  return '{:.1f}s'.format(t)


def compute_acc(pred, label, reduction='mean'):
  result = (pred == label).float()
  if reduction == 'none':
    return result.detach()
  elif reduction == 'mean':
    return result.mean().item()


def compute_n_params(model, return_str=True):
  n_params = 0
  for p in model.parameters():
    n_params += p.numel()
  if return_str:
    if n_params >= 1e6:
      return '{:.1f}M'.format(n_params / 1e6)
    else:
      return '{:.1f}K'.format(n_params / 1e3)
  else:
    return n_params


def mean_confidence_interval(data, confidence=0.95):
  a = 1.0 * np.array(data)
  stderr = stats.sem(a)
  h = stderr * stats.t.ppf((1 + confidence) / 2., len(a) - 1)
  return h


def config_inner_args(inner_args):
  if inner_args is None: 
    inner_args = dict()

  inner_args['reset_classifier'] = inner_args.get('reset_classifier') or False
  inner_args['n_step'] = inner_args.get('n_step') or 5
  inner_args['encoder_lr'] = inner_args.get('encoder_lr') or 0.01
  inner_args['classifier_lr'] = inner_args.get('classifier_lr') or 0.01
  inner_args['momentum'] = inner_args.get('momentum') or 0.
  inner_args['weight_decay'] = inner_args.get('weight_decay') or 0.
  inner_args['first_order'] = inner_args.get('first_order') or False
  inner_args['frozen'] = inner_args.get('frozen') or []

  return inner_args