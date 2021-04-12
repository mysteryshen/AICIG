# Adversarial Imbalance Classification with Class-specific Diverse Instance Generation (AICIG)
This is the official implementation of AICIG.
## Abstract
Data augmentation is an effective technique for imbalance classification. However, it still suffers from two key issues. Firstly, data augmentation and classifier construction are performed separately, where classifier construction may not
benefit from the augmentation strategies. Secondly, low variations in generated instances may lead to overfitting problem. In this paper, an Adversarial Imbalance Classification method with
Class-specific Diverse Instance Generation is proposed (AICIG). It is a framework that unifies adversarial classifier construction and class-specific diverse instance generation, where these two
stages enforce each other seamlessly. Concretely, a specially designed adversarial classifier drives the generation of class-specific instances by class-specific adversarial process. Meanwhile, the
distribution of latent variables is assumed to be a Gaussian mixture, which enables to generate diverse instances with limited data. With augmented class-specific and diverse instances, the adversarial classifier can obtain better generalization performance.
We conduct experiments on four widely-used imbalanced image datasets and compare them with the state-of-the-art methods. The experimental results exhibit that our method can effectively
prevent overfitting and obtain better performance on imbalance classification tasks.  
![image](https://github.com/mysteryshen/AICIG/blob/master/model.pdf)
## Requirement
The code was tested on:
* python=3.6
* tensorflow=1.15.0 (gpu version)
* keras=2.2.4
* torchvision=0.5.0 (utilized for dataset preparation)
## Usage
`python train_dataname.py`  
`eg: python train_mnist.py`  
The dataset will be automatically downloaded and prepared in `./dataset/` when first run.
## License
MIT
