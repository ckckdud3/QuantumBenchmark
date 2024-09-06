import pennylane as qml
import torch
import pickle
import logging
import time
import argparse
import matplotlib.pyplot as plt

from qcnn.core.qcnn import QCNNSequential, scheme_builder
from qcnn.core.blocks import *
from qcnn.core.utils.utils import *
from qcnn.core.utils.customparser import customparser


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('config_file_name', help='A yaml file name which contains configurations (e.g. config.yaml)')
arg = parser.parse_args()


custom_parser = customparser(arg.config_file_name)    
parsed_args = custom_parser.parse_custom_args()

num_wires = parsed_args[0].num_wires

batch_size = parsed_args[1].batch_size
epoch = parsed_args[1].num_epoch

dataset_path = parsed_args[2].dataset_path
save_to = parsed_args[2].save_to


dev = qml.device('default.qubit', wires=num_wires)


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler(f'{save_to}.log', mode='w')
                    ])
logger = logging.getLogger('main')


block_list = scheme_builder(parsed_args[0])

qcnn = QCNNSequential(block_list, dev, num_wires)

# qcnn = QCNNSequential([QCNNEmbeddingBlock([0,1,2],1), QCNNEmbeddingBlock([3,4,5],1),
#                        QCNNCrossConvBlock([0,1,2],[3,4,5],3), QCNNCrossConvBlock([1,2],[4,5],3)], dev, num_wires)

qcnn.draw_circuit()

dataset=None
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)
    f.close()

npdataset = []

for i in range(len(dataset)):
    npdataset.append(np.array(dataset[i]))

train_x, train_y = dataset_shuffle(npdataset[0], npdataset[1])
test_x, test_y = dataset_shuffle(npdataset[2], npdataset[3])


outdata = {}
cost_data = []
accuracy_per_epoch = []

logger.info(f'Training started : Batch size = {batch_size}, epoch = {epoch}')
start = time.time()

for e in range(epoch):

    if parsed_args[1].shuffle_per_epoch:
        del train_x, train_y
        train_x, train_y = dataset_shuffle(npdataset[0], npdataset[1])

    logger.info(f'Epoch #{e+1}')

    for i in range(int(len(train_x)/batch_size)):

        logger.info(f'Iteration #{i+1}')
        cost = qcnn.step(train_x[i*batch_size:(i+1)*batch_size], train_y[i*batch_size:(i+1)*batch_size])

        logger.info(f'Cost : {cost}')
        cost_data.append(cost)

    logger.info(f'Epoch #{e+1} Test')

    total = len(test_x)

    data = test_x
    label = torch.tensor(test_y).to('cuda:0')

    eval = qcnn(data)
    count = torch.sum(torch.sign(label) == torch.sign(eval))
    logger.info(f'Epoch #{e+1} Accuracy = {count/total}')
    accuracy_per_epoch.append(count.item()/total)


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
outdata['num_processor'] = parsed_args[0].num_processor
outdata['elapsed_time'] = seconds
outdata['num_epoch'] = epoch
outdata['batch_size'] = batch_size
outdata['cost_data'] = cost_data
outdata['accuracy_per_epoch'] = accuracy_per_epoch
outdata['parameters'] = qcnn.get_weights()


logger.info('Test started')

total = len(test_x)

data = test_x
label = torch.tensor(test_y).to('cuda:0')

eval = qcnn(data)
count = torch.sum(torch.sign(label) == torch.sign(eval))

logger.info(f'Accuracy = {count/total}')

outdata['test_accuracy'] = (count/total
).item()

with open(save_to+'_traindata.pickle', 'wb') as f:
    pickle.dump(outdata, f)
    f.close()

# plot_cost_accuracy(save_to)