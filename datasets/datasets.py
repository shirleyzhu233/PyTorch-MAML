import os

import torch


DEFAULT_ROOT = './materials'
datasets = {}

def register(name):
  def decorator(cls):
    datasets[name] = cls
    return cls
  return decorator


def make(name, **kwargs):
  if kwargs.get('root_path') is None:
    kwargs['root_path'] = os.path.join(DEFAULT_ROOT, name.replace('meta-', ''))
  dataset = datasets[name](**kwargs)
  return dataset


def collate_fn(batch):
  shot, query, shot_label, query_label = [], [], [], []
  for s, q, sl, ql in batch:
    shot.append(s)
    query.append(q)
    shot_label.append(sl)
    query_label.append(ql)
  
  shot = torch.stack(shot)                # [n_ep, n_way * n_shot, C, H, W]
  query = torch.stack(query)              # [n_ep, n_way * n_query, C, H, W]
  shot_label = torch.stack(shot_label)    # [n_ep, n_way * n_shot]
  query_label = torch.stack(query_label)  # [n_ep, n_way * n_query]
  
  return shot, query, shot_label, query_label