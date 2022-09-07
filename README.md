# CurAML
Code of the T-PAMI paper "Curvature-Adaptive Meta-Learning for Fast Adaptation to Manifold Data"

Here we provide the code of training our model on the Mini-ImageNet dataset.

1. Download the images of the Mini-ImageNet dataset.

2. The folder 'pretrain' is used to pretrain the backbone of a product manifold neural network. Training data is in the 'materials' folder.
You can run
-------
```
python train_classifier.py 
```

3. The folder 'meta-learning' is used for meta-learning. You can run 
-------
```
python miniimagenet_train.py --k_spt 5 --meta_lr 5e-5 --update_lr 2e-1 --k_lr 1e-5
```
or
```
python miniimagenet_train.py --k_spt 1 --meta_lr 5e-5 --update_lr 2e-1 --k_lr 1e-5
```
