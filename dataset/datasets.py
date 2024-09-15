import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data.dataset import TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np


def get_filtered_indices(dataset, labels, n, n_val=0):
        
        indices, l = [], []
        for i, label in enumerate(dataset.targets):
            if label in labels:
                indices.append(i)
                l.append(label)

        if n_val == 0:
            _, ret = train_test_split(
                indices,
                test_size = n,
                random_state = 42,
                stratify = l
            )

            return ret
        
        else:

            ret_train, ret_val = train_test_split(
                indices,
                train_size = n,
                test_size = n_val,
                random_state = 42,
                stratify = l
            )

            return ret_train, ret_val


def dataset_to_numpy(dataset):

    data = []
    targets = []

    for img, target in dataset:
        data.append(img.numpy().flatten())
        targets.append(target)

    return np.array(data), np.array(targets)


def get_pca_dataset(n_dim, n_train, n_val, n_test):

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    train_idx, val_idx = get_filtered_indices(train_dataset, [0, 1], n_train, n_val)
    test_idx = get_filtered_indices(test_dataset, [0, 1], n_test)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)

    train_data, train_targets = dataset_to_numpy(train_subset)
    val_data, val_targets = dataset_to_numpy(val_subset)
    test_data, test_targets = dataset_to_numpy(test_subset)


    pca = PCA(n_components = n_dim)
    train_data_pca = pca.fit_transform(train_data)
    val_data_pca = pca.transform(val_data)
    test_data_pca = pca.transform(test_data)


    train_data_pca = torch.tensor(train_data_pca, dtype=torch.float64)
    train_targets = torch.tensor(train_targets, dtype=torch.long)
    val_data_pca = torch.tensor(val_data_pca, dtype=torch.float64)
    val_targets = torch.tensor(val_targets, dtype=torch.long)
    test_data_pca = torch.tensor(test_data_pca, dtype=torch.float64)
    test_targets = torch.tensor(test_targets, dtype=torch.long)


    train_dataset_pca = TensorDataset(train_data_pca, train_targets)
    val_dataset_pca = TensorDataset(val_data_pca, val_targets)
    test_dataset_pca = TensorDataset(test_data_pca, test_targets)

    return train_dataset_pca, val_dataset_pca, test_dataset_pca