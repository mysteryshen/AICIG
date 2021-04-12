import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from CelebA import CelebA
import os

_torch_supported_dataset = ('mnist', 'fashion', 'cifar10', 'svhn')
_custom_dataset = {'gtsrb', 'celeba'}
_torch_dataset_key_mapping = {
    'mnist': 'MNIST',
    'fashion': 'FashionMNIST',
    'cifar10': 'CIFAR10',
    'svhn': 'SVHN',
    'celeba': 'CelebA',
}
_originally_imbalance_dataset = {'gtsrb'}
_dataset_ratio_mapping = {
    'mnist': [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40],
    'fashion': [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40],
    'cifar10': [4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80],
    'svhn': [4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80],
    'celeba': [15000, 1500, 750, 300, 150]
}

def dataset_to_numpy(dataset):
    loader = DataLoader(dataset, len(dataset))
    x, y = next(iter(loader))
    return x.numpy(), y.numpy()

def load_data(name, seed, imbalance=None, data_dir=None):
    name = name.lower()
    if data_dir is None:
        data_dir = './dataset/%s/' % name
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    if name in _torch_supported_dataset:
        func_name = _torch_dataset_key_mapping[name]  # if name in _torch_dataset_key_mapping else None

        dataset_func = getattr(torchvision.datasets, func_name)
        transform = transforms.Compose([transforms.ToTensor(), ])
        if name in ('mnist', 'fashion', 'cifar10'):
            train_dataset = dataset_func(data_dir, train=True, transform=transform, download=True)
            test_dataset = dataset_func(data_dir, train=False, transform=transform, download=True)
        elif name == 'svhn':
            train_dataset = dataset_func(data_dir, split='train', transform=transform, download=True)
            test_dataset = dataset_func(data_dir, split='test', transform=transform, download=True)

        else:
            raise NotImplementedError
    elif name in _custom_dataset:
        if name == 'gtsrb':
            train_dataset = GTSRB(data_dir, train=True, download=True)
            test_dataset = GTSRB(data_dir, train=False, download=True)
        elif name == 'celeba':
            train_dataset = CelebA(data_dir, split='train', download=True)
            test_dataset = CelebA(data_dir, split='test', download=True)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)
    X_train = 2*X_train - 1#normalize to [-1,1]
    X_test = 2*X_test - 1
    X_train, y_train = _shuffle(X_train, y_train, seed)
    X_train = np.transpose(X_train, axes=[0, 2, 3, 1])
    X_test = np.transpose(X_test, axes=[0, 2, 3, 1])
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    if imbalance is None or imbalance is False or name in _originally_imbalance_dataset:
        return (X_train, y_train), (X_test, y_test)
    if imbalance is True:
        ratio = _dataset_ratio_mapping[name]
    else:
        ratio = imbalance
    X_train = [X_train[y_train == i][:num] for i, num in enumerate(ratio)]
    y_train = [y_train[y_train == i][:num] for i, num in enumerate(ratio)]
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_train, y_train = _shuffle(X_train, y_train, seed)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return (X_train, y_train), (X_test, y_test)


def _shuffle(x, y, seed):
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    return x, y


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ((x, y), (x_test, y_test)) = load_data('cifar10', 0, imbalance=True)
    n_classes = len(np.unique(y))
    for i in range(n_classes):
        x_plot = x[y == i][:15]
        for j in range(len(x_plot)):
            plt.imsave("img/class%d_%d.png" % (i, j), x_plot[j])
