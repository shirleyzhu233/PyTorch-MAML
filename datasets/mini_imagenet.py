import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import get_transform


@register('mini-imagenet')
class MiniImageNet(Dataset):
  def __init__(self, root_path, split='train', image_size=84, 
               normalization=True, transform=None):
    super(MiniImageNet, self).__init__()
    split_dict = {'train': 'train_phase_train',        # standard train
                  'val': 'train_phase_val',            # standard val
                  'trainval': 'train_phase_trainval',  # standard train and val
                  'test': 'train_phase_test',          # standard test
                  'meta-train': 'train_phase_train',   # meta-train
                  'meta-val': 'val',                   # meta-val
                  'meta-test': 'test',                 # meta-test
                 }
    split_tag = split_dict[split]

    split_file = os.path.join(root_path, split_tag + '.pickle')
    assert os.path.isfile(split_file)
    with open(split_file, 'rb') as f:
      pack = pickle.load(f, encoding='latin1')
    data, label = pack['data'], pack['labels']

    data = [Image.fromarray(x) for x in data]
    label = np.array(label)
    label_key = sorted(np.unique(label))
    label_map = dict(zip(label_key, range(len(label_key))))
    new_label = np.array([label_map[x] for x in label])
    
    self.root_path = root_path
    self.split_tag = split_tag
    self.image_size = image_size

    self.data = data
    self.label = new_label
    self.n_classes = len(label_key)

    if normalization:
      self.norm_params = {'mean': [0.471, 0.450, 0.403],
                          'std':  [0.278, 0.268, 0.284]}
    else:
      self.norm_params = {'mean': [0., 0., 0.],
                          'std':  [1., 1., 1.]}

    self.transform = get_transform(transform, image_size, self.norm_params)

    def convert_raw(x):
      mean = torch.tensor(self.norm_params['mean']).view(3, 1, 1).type_as(x)
      std = torch.tensor(self.norm_params['std']).view(3, 1, 1).type_as(x)
      return x * std + mean
    
    self.convert_raw = convert_raw

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    image = self.transform(self.data[index])
    label = self.label[index]
    return image, label


@register('meta-mini-imagenet')
class MetaMiniImageNet(MiniImageNet):
  def __init__(self, root_path, split='train', image_size=84, 
               normalization=True, transform=None, val_transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15):
    super(MetaMiniImageNet, self).__init__(root_path, split, image_size, 
                                           normalization, transform)
    self.n_batch = n_batch
    self.n_episode = n_episode
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_query = n_query

    self.catlocs = tuple()
    for cat in range(self.n_classes):
      self.catlocs += (np.argwhere(self.label == cat).reshape(-1),)

    self.val_transform = get_transform(
      val_transform, image_size, self.norm_params)

  def __len__(self):
    return self.n_batch * self.n_episode

  def __getitem__(self, index):
    shot, query = [], []
    cats = np.random.choice(self.n_classes, self.n_way, replace=False)
    for c in cats:
      c_shot, c_query = [], []
      idx_list = np.random.choice(
        self.catlocs[c], self.n_shot + self.n_query, replace=False)
      shot_idx, query_idx = idx_list[:self.n_shot], idx_list[-self.n_query:]
      for idx in shot_idx:
        c_shot.append(self.transform(self.data[idx]))
      for idx in query_idx:
        c_query.append(self.val_transform(self.data[idx]))
      shot.append(torch.stack(c_shot))
      query.append(torch.stack(c_query))
    
    shot = torch.cat(shot, dim=0)             # [n_way * n_shot, C, H, W]
    query = torch.cat(query, dim=0)           # [n_way * n_query, C, H, W]
    cls = torch.arange(self.n_way)[:, None]
    shot_labels = cls.repeat(1, self.n_shot).flatten()    # [n_way * n_shot]
    query_labels = cls.repeat(1, self.n_query).flatten()  # [n_way * n_query]
    
    return shot, query, shot_labels, query_labels