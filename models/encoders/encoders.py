import torch


models = {}

def register(name):
  def decorator(cls):
    models[name] = cls
    return cls
  return decorator


def make(name, **kwargs):
  if name is None:
    return None
  model = models[name](**kwargs)
  if torch.cuda.is_available():
    model.cuda()
  return model


def load(ckpt):
  model = make(ckpt['encoder'], **ckpt['encoder_args'])
  if model is not None:
    model.load_state_dict(ckpt['encoder_state_dict'])
  return model