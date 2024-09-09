import pennylane as qml
import torch
from torch.utils.data.dataloader import DataLoader
import pickle
import logging
import time
import argparse
import matplotlib.pyplot as plt

from qcnn.core.qcnn import QCNNSequential, scheme_builder
from qcnn.core.blocks import *
from qcnn.core.utils.utils import *
from qcnn.core.utils.customparser import customparser

from dataset.datasets import get_pca_dataset

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('config_file_name', help='A yaml file name which contains configurations (e.g. config.yaml)')
arg = parser.parse_args()

train_size = 2000
test_size = 500

data_dim = 12

custom_parser = customparser(arg.config_file_name)    
parsed_args = custom_parser.parse_custom_args()

num_wires = parsed_args[0].num_wires
out_dim = parsed_args[0].out_dim

batch_size = parsed_args[1].batch_size
epoch = parsed_args[1].num_epoch

dataset_path = parsed_args[2].dataset_path
save_to = parsed_args[2].save_to

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dev = qml.device('default.qubit', wires=num_wires)


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler(f'{save_to}.log', mode='w')
                    ])
logger = logging.getLogger('main')


block_list = scheme_builder(parsed_args[0], data_dim)

qcnn = QCNNSequential(block_list, dev, num_wires, out_dim)

qcnn.draw_circuit(save_to)

print(len(qcnn.get_weights()))

train_dataset, test_dataset = get_pca_dataset(data_dim, train_size, test_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

outdata = {}
cost_data = []
accuracy_per_epoch = []

logger.info(f'Training started : Batch size = {batch_size}, epoch = {epoch}')
start = time.time()


for e in range(epoch):

    logger.info(f'Epoch #{e+1}')

    for idx, (data, label) in enumerate(train_loader):

        data, label = data.to(device), label.to(device)
        logger.info(f'Iteration #{idx+1}')
        cost = qcnn.step(data, label)

        logger.info(f'Cost : {cost}')
        cost_data.append(cost)
        pass

    logger.info(f'Epoch #{e+1} Test')

    count = 0
    for idx, (data, label) in enumerate(test_loader):

        data, label = data.to(device), label.to(device)
        eval = qcnn(data)
        count += torch.sum(torch.max(eval,axis=1).indices == label).item()

    logger.info(f'Epoch #{e+1} Accuracy : {count/test_size}')
    accuracy_per_epoch.append(count/test_size)


end = time.time()

seconds = int(end - start)
modseconds = (seconds) % 60
minutes = (seconds // 60) % 60
hours = (seconds // 3600) % 24
days = (seconds // 86400)

logger.info(f'Training ended. Elapsed time: {days} days {hours} hours {minutes} minutes {modseconds} seconds')

outdata['scheme'] = parsed_args[0].scheme
outdata['depth'] = parsed_args[0].depth
outdata['num_qubit'] = parsed_args[0].num_wires
outdata['elapsed_time'] = seconds
outdata['num_epoch'] = epoch
outdata['batch_size'] = batch_size
outdata['cost_data'] = cost_data
outdata['accuracy_per_epoch'] = accuracy_per_epoch
outdata['parameters'] = qcnn.get_weights()
outdata['num_parameters'] = len(qcnn.get_weights())


logger.info('Test started')


count = 0
for idx, (data, label) in enumerate(test_loader):

    data, label = data.to(device), label.to(device)
    eval = qcnn(data)
    count += torch.sum(torch.max(eval,axis=1).indices == label).item()


logger.info(f'Accuracy : {count/test_size}')
accuracy_per_epoch.append(count/test_size)

outdata['test_accuracy'] = (count/test_size)

with open(save_to+'_traindata.pickle', 'wb') as f:
    pickle.dump(outdata, f)
    f.close()

# plot_cost_accuracy(save_to)