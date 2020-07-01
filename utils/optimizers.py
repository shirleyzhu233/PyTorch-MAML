from torch.optim import SGD, RMSprop, Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


def make(name, params, lr, weight_decay=0., 
         schedule='step', milestones=None, gamma=0.1):
  """
  Prepares an optimizer and its learning-rate scheduler.

  Args:
    name (str): name of the optimizer. Options: 'sgd', 'rmsprop', 'adam'
    params (iterable): parameters to optimize.
    lr (float): initial learning rate.
    weight_decay (float, optional): weight decay. Default: 0.
    schedule (str, optional): type of learning-rate schedule. Default: 'step'
      Options: 'step', 'cosine'
      (This argument is ignored if milestones=None.)
    milestones (int list, optional): a list of epoches when learning rate 
      is altered. Default: None
    gamma (float, optional): multiplicative factor of learning rate decay.
      Default: 0.1
  """
  if name == 'sgd':
    optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
  elif name == 'rmsprop':
    optimizer = RMSprop(params, lr, weight_decay=weight_decay)
  elif name == 'adam':
    optimizer = Adam(params, lr, weight_decay=weight_decay)
  else:
    raise ValueError('invalid optimizer')
  
  if milestones is not None:
    if schedule == 'step':
      lr_scheduler = MultiStepLR(optimizer, milestones, gamma)
    elif schedule == 'cosine':
      lr_scheduler = CosineAnnealingLR(optimizer, milestones[-1])
  else:
    lr_scheduler = None
  
  return optimizer, lr_scheduler


def load(ckpt, params):
  train = ckpt['training']
  optimizer, lr_scheduler = make(
    train['optimizer'], params, **train['optimizer_args'])
  optimizer.load_state_dict(train['optimizer_state_dict'])
  
  if lr_scheduler is not None:
    lr_scheduler.load_state_dict(train['lr_scheduler_state_dict'])
  
  return optimizer, lr_scheduler