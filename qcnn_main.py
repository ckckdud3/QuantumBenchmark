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
val_size = 500
test_size = 500

custom_path = False

custom_parser = customparser(arg.config_file_name)    
parsed_args = custom_parser.parse_custom_args()

num_wires = parsed_args[0].num_wires
out_dim = parsed_args[0].out_dim

batch_size = parsed_args[1].batch_size
epoch = parsed_args[1].num_epoch
data_dim = parsed_args[1].data_dim

if not custom_path:
    filename = f'{parsed_args[0].num_wires}_{parsed_args[0].scheme}_{parsed_args[0].embed_type}_{parsed_args[0].block_type}_{parsed_args[0].num_processor}processor_depth{parsed_args[0].depth}'
else:
    filename = parsed_args[2].save_to

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dev = qml.device('default.qubit', wires=num_wires)


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler(f'{filename}.log', mode='w')
                    ])
logger = logging.getLogger('main')


block_list = scheme_builder(parsed_args[0], data_dim)

qcnn = QCNNSequential(block_list, dev, parsed_args[0])

qcnn.draw_circuit(filename)

train_dataset, val_dataset, test_dataset = get_pca_dataset(data_dim, train_size, val_size, test_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

outdata = {}
cost_data = []
var_data = []
grad_norm_data = []
accuracy_per_epoch = []

logger.info(f'Training started : Batch size = {batch_size}, epoch = {epoch}')
start = time.time()


for e in range(epoch):

    logger.info(f'Epoch #{e+1}')

    for idx, (data, label) in enumerate(train_loader):

        data, label = data.to(device), label.to(device)
        logger.info(f'Iteration #{idx+1}')
        cost, var, norm = qcnn.step(data, label)

        logger.info(f'Cost : {cost}, Grad [Variance : {var}, Norm : {norm}]')
        cost_data.append(cost)
        var_data.append(var)
        grad_norm_data.append(norm)
        pass

    logger.info(f'Epoch #{e+1} Validation')

    count = 0
    for idx, (data, label) in enumerate(val_loader):

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


outdata['data_dim'] = parsed_args[1].data_dim
outdata['scheme'] = parsed_args[0].scheme
outdata['block_type'] = parsed_args[0].block_type
outdata['depth'] = parsed_args[0].depth
outdata['num_qubit'] = parsed_args[0].num_wires
outdata['elapsed_time'] = seconds
outdata['num_epoch'] = epoch
outdata['batch_size'] = batch_size
outdata['cost_data'] = cost_data
outdata['accuracy_per_epoch'] = accuracy_per_epoch
outdata['grad_norm'] = grad_norm_data
outdata['var_data'] = var_data
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

with open(filename + '_traindata.pickle', 'wb') as f:
    pickle.dump(outdata, f)
    f.close()

# plot_cost_accuracy(save_to)