import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import numpy as np


def get_filtered_indices(dataset, labels):
        indices = [i for i, label in enumerate(dataset.targets) if label in labels]
        return indices


def dataset_to_numpy(dataset):
    data = []
    targets = []
    for img, target in dataset:
        data.append(img.numpy().flatten())
        targets.append(target)
    return np.array(data), np.array(targets)


def get_pca_dataset(n_dim = 20):

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    train_indices = get_filtered_indices(train_dataset, [0, 1])
    test_indices = get_filtered_indices(test_dataset, [0, 1])


    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)


    train_data, train_targets = dataset_to_numpy(train_subset)
    test_data, test_targets = dataset_to_numpy(test_subset)


    pca = PCA(n_components = n_dim)
    train_data_pca = pca.fit_transform(train_data)
    test_data_pca = pca.transform(test_data)


    train_data_pca = torch.tensor(train_data_pca, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.long)
    test_data_pca = torch.tensor(test_data_pca, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.long)


    train_dataset_pca = torch.utils.data.TensorDataset(train_data_pca, train_targets)
    test_dataset_pca = torch.utils.data.TensorDataset(test_data_pca, test_targets)

    return train_dataset_pca, test_dataset_pca