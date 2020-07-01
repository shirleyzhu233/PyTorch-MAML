import torchvision.transforms as transforms


__all__ = ['get_transform']


def get_transform(name, image_size, norm_params):
  if name == 'resize':
    return transforms.Compose([
      transforms.RandomResizedCrop(image_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**norm_params),
    ])
  elif name == 'crop':
    return transforms.Compose([
      transforms.Resize(image_size),
      transforms.RandomCrop(image_size, padding=8),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**norm_params),
    ])
  elif name == 'color':
    return transforms.Compose([
      transforms.Resize(image_size),
      transforms.RandomCrop(image_size, padding=8),
      transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**norm_params),
    ])
  elif name == 'flip':
    return transforms.Compose([
      transforms.Resize(image_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**norm_params),
    ])
  elif name == 'enlarge':
    return transforms.Compose([
      transforms.Resize(int(image_size * 256 / 224)),
      transforms.CenterCrop(image_size),
      transforms.ToTensor(),
      transforms.Normalize(**norm_params),
    ])
  elif name is None:
    return transforms.Compose([
      transforms.Resize(image_size),
      transforms.ToTensor(),
      transforms.Normalize(**norm_params),
    ])
  else:
    raise ValueError('invalid transformation')