import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np


def get_filtered_indices(dataset, labels, n):
        indices, l = [], []
        for i, label in enumerate(dataset.targets):
            if label in labels:
                indices.append(i)
                l.append(label)

        _, ret = train_test_split(
            indices,
            test_size = n,
            random_state = 42,
            stratify = l
        )
        return ret


def dataset_to_numpy(dataset):
    data = []
    targets = []
    for img, target in dataset:
        data.append(img.numpy().flatten())
        targets.append(target)
    return np.array(data), np.array(targets)


def get_pca_dataset(n_dim, n_train, n_test):

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    train_idx = get_filtered_indices(train_dataset, [0, 1], n_train)
    test_idx = get_filtered_indices(test_dataset, [0, 1], n_test)

    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    test_subset = torch.utils.data.Subset(test_dataset, test_idx)

    train_data, train_targets = dataset_to_numpy(train_subset)
    test_data, test_targets = dataset_to_numpy(test_subset)


    pca = PCA(n_components = n_dim)
    train_data_pca = pca.fit_transform(train_data)
    test_data_pca = pca.transform(test_data)


    train_data_pca = torch.tensor(train_data_pca, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.long)
    test_data_pca = torch.tensor(test_data_pca, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.long)


    train_dataset_pca = Dataset(train_data_pca, train_targets)
    test_dataset_pca = Dataset(test_data_pca, test_targets)

    return train_dataset_pca, test_dataset_pca