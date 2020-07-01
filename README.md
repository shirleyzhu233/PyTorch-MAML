# MAML in PyTorch - Re-implementation and Beyond

A PyTorch implementation of [Model Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400). We faithfully reproduce [the official Tensorflow implementation](https://github.com/cbfinn/maml) while incorporating a number of additional features that may ease further study of this very high-profile meta-learning framework. 

## Overview

This repository contains code for training and evaluating MAML on the mini-ImageNet and tiered-ImageNet datasets most commonly used for few-shot image classification. To the best of our knowledge, this is the only PyTorch implementation of MAML to date that **fully reproduces the results in the original paper** without applying tricks such as data augmentation, evaluation on multiple crops, and ensemble of multiple models. Other existing PyTorch implementations typically see a ~3% gap in accuracy for the 5-way-1-shot and 5-way-5-shot classification tasks on mini-ImageNet.

Beyond reproducing the results, our implementation comes with a few extra bits that we believe can be helpful for further development of the framework. We highlight the improvements we have built into our code, and discuss our observations that warrent some attention.

## Implementation Highlights

- **Batch normalization with per-episode running statistics.** Our implementation provides flexibility of tracking global and/or per-episode running statistics, hence supporting both transductive and inductive inference.

- **Better data pre-processing.** The official implementation does not normalize and augment data. We support data normalization and a variety of data augmentation techniques. We also implement data batching and support/query-set splitting more efficiently.

- **More datasets.** We support mini-ImageNet, tiered-ImageNet and more.

- **More options for outer-loop optimization.** We support mutiple optimizers and learning-rate schedulers for the outer-loop optimization.

- **More powerful inner-loop optimization.** The official implementation uses vanilla gradient descent in the inner loop. We support momentum and weight decay.

- **More options for encoder architecture.** We support the standard four-layer ConvNet as well as ResNet-12 and ResNet-18 as the encoder.

- **Easy layer freezing.** We provide an interface for layer freezing experiments. One may freeze an arbitrary set of layers or blocks during inner-loop adaptation.

- **Meta-learning with zero-initialized classifier head.** The official implementation learns a meta-initialization for both the encoder and the classifier head. This prevents one from varying the number of categories at training or test time. With our implementation, one may opt to learn a meta-initialization for the encoder while initializing the classifier head to zero.

- **Distributed training and gradient checkpointing.** MAML is very memory-intensive because it buffers all tensors generated throughout the inner-loop adaptation steps. Gradient checkpointing trades compute for memory, effectively bringing the memory cost from O(N) down to O(1), where N is the number of inner-loop steps. In our experiments, gradient checkpointing saves up to 80% of GPU memory at the cost of running the forward pass more than once (a moderate 20% increase in running time).

## Transductive or Inductive?

The official implementation assumes transductive learning. The batch normalization layers do not track running statistics at training time, and they use mini-batch statistics at test time. The implicit assumption here is that test data come in mini-batches and are perhaps balanced across categories. This is a very restrictive assumption and does not land MAML directly comparable with the vast majority of meta-learning and few-shot learning methods. Unfortunately, this is not immediately obvious from the paper, and our findings suggest that the performance of MAML is hugely overestimated.

- **Accuracy is very sensitive to the size of query set in the transductive setting.** For example, the result for 5-way-1-shot classification on miniImageNet from the paper (48.70%) was obtained on five queries, one per category. We found that the accuracy dropped by ~1.5% given five queries per category, and by ~2.5% given 15 queries per category.

- The paper reports mean accuracy over 600 independently sampled tasks, or trials. We found that **600 trials, again in the transductive setting, are insufficient for an unbiased estimate of model performance**. The mean accuracy from 6,000 trails is more stable, and is always ~2% lower than that from the first 600 trials. We conjecture that the distribution of per-trial accuracy is highly skewed toward the high end.

- We found that **MAML performs a lot worse in the inductive setting**. Given the same model configuration, inductive accuracy is always much lower (~4%) than the *corrected* transductive accuracy, which is already a few percentage points behind the reported number.

Hence, one should be extremely cautious when comparing MAML with its competitors as is evident from the discussion above.

## FOMAML and layer freezing

Unfortunately, some insights discussed in the original paper and its follow-up works do not hold in the inductive setting. 

- FOMAML (i.e. the first-order approximation of MAML) performs as well as MAML in transductive learning, but fails completely in the inductive setting. 

- Completely freezing the encoder during inner-loop adaption as was done in [this work](https://arxiv.org/abs/1909.09157) results in dramatic decrease in accuracy.

## BatchNorm and TaskNorm

[A recent work](https://arxiv.org/abs/2003.03284) proposes a test-time enhancement of batchnorms, noting that the small batch sizes during training may leave batch normalization less effective. We did not have much success with this method. We observed marginal improvement most of the time, and found that it hurt performance occationally. That said, we do believe that batch normalization is hard to deal with in MAML. TaskNorm attempts to attack the problem of small batch sizes, which we conjecture is just one among the three causes (i.e., extremely scarse training data, extremely small batch sizes, and extremely small number of inner-loop updates) of the ineffectiveness of batch normalization in MAML.

## Quick Start

### 0. Preliminaries

**Environment**

- Python 3.6.8 (or any Python 3 distribution)
- PyTorch 1.3.1 (or any PyTorch > 1.0)
- tensorboardX

**Datasets**

Please follow the download links [here](https://github.com/cyvius96/few-shot-meta-baseline). Please modify the file names accordingly so that they can be recognized by the data loaders.

**Configurations**

Template configuration files as well as those for reproducing the results in the original paper can be found in `configs/`. The hyperparameters are self-explanatory.

### 1. Training MAML
Here is the command for single-GPU training of MAML with ConvNet4 backbone for 5-way-1-shot classification on mini-ImageNet to reproduce the result in the original paper.
```
python train.py --config=configs/convnet4/mini-imagenet/train_reproduce.yaml
```

Use `-gpu` to specify available GPUs for multi-GPU training. For example,
```
python train.py --config=configs/convnet4/mini-imagenet/train_reproduce.yaml --gpu=0,1
```

Add '-efficient' to enable gradient checkpointing. This aggresively saves GPU memory while slightly increases running time.
```
python train.py --config=configs/convnet4/mini-imagenet/train_reproduce.yaml --efficient
```

Use '-tag' to customize the name of the directory where the checkpoints and log files are saved.

### 2. Testing MAML
Here is how one would test MAML for 5-way-1-shot classification on mini-ImageNet to reproduce the result in the original paper. Please confirm the loading path first.
```
python test.py --config=configs/convnet4/mini-imagenet/test_reproduce.yaml
```

The `-gpu` and `-efficient` tags function similarly as in training.

## Contact
[Fangzhou Mu](http://pages.cs.wisc.edu/~fmu/) (fmu2@wisc.edu)

## Cite our Repository
```
@misc{pytorch_maml,
  author = {Fangzhou Mu},
  title = {maml in pytorch - re-implementation and beyond},
  year = {2020},
  howpublished = {\url{https://github.com/fmu2/PyTorch-MAML}},
}
```

## Related Code Repositories

Our implementation is inspired by the following repositories.
* maml (the official implementation) <https://github.com/cbfinn/maml>
* MAML-Pytorch <https://github.com/dragen1860/MAML-Pytorch>
* HowToTrainYourMAMLPytorch <https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch>
* memory-efficient-maml <https://github.com/dbaranchuk/memory-efficient-maml>

## References
```
@inproceedings{finn2017model,
  title={Model-agnostic meta-learning for fast adaptation of deep networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2017}
}

@inproceedings{raghu2019rapid,
  title={Rapid learning or feature reuse? towards understanding the effectiveness of maml},
  author={Raghu, Aniruddh and Raghu, Maithra and Bengio, Samy and Vinyals, Oriol},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019}
}

@article{Bronskill2020tasknorm,
  title={Tasknorm: rethinking batch normalization for meta-learning},
  author={Bronskill, John and Gordon, Jonathan and Requeima, James and Nowozin, Sebastian and Turner, Richard E.},
  journal={arXiv preprint arXiv:2003.03284},
  year={2020}
}
```