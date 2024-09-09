import numpy as np
import pickle
import matplotlib.pyplot as plt

def dataset_shuffle(data, label):

    vecsize = data.shape[1]
    integrated = np.hstack((data, label.reshape(-1,1)))

    np.random.shuffle(integrated)

    return integrated[:,:vecsize], integrated[:,-1]


def open_file(filename: str):
    ret = None
    with open(filename, 'rb') as f:
        ret = pickle.load(f)
        f.close()

    return ret


def plot_cost_accuracy(save_to):

    plt.figure(figsize=(8,6))

    plt.clf()

    train_data = open_file(f'{save_to}_traindata.pickle')

    ape = train_data['accuracy_per_epoch']

    cost_x = np.arange(len(train_data['cost_data']))

    plt.plot(cost_x, train_data['cost_data'])

    plt.savefig(f'{save_to}_cost.png')

    plt.clf()

    ape_x = np.arange(len(ape))
    plt.plot(ape_x, ape)

    plt.savefig(f'{save_to}_accuracy.png')