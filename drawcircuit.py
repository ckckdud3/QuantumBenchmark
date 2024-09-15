import pennylane as qml
import torch
import argparse

from qcnn.core.qcnn import QCNNSequential, scheme_builder
from qcnn.core.blocks import *
from qcnn.core.utils.utils import *
from qcnn.core.utils.customparser import customparser

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('config_file_name', help='A yaml file name which contains configurations (e.g. config.yaml)')
arg = parser.parse_args()

custom_path = False

custom_parser = customparser(arg.config_file_name)    
parsed_args = custom_parser.parse_custom_args()

num_wires = parsed_args[0].num_wires
data_dim = parsed_args[1].data_dim

if not custom_path:
    filename = f'{parsed_args[0].num_wires}_{parsed_args[0].scheme}_{parsed_args[0].embed_type}_{parsed_args[0].block_type}_depth{parsed_args[0].depth}'
else:
    filename = parsed_args[2].save_to

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dev = qml.device('default.qubit', wires=num_wires)

block_list = scheme_builder(parsed_args[0], data_dim)

qcnn = QCNNSequential(block_list, dev, parsed_args[0])

qcnn.draw_circuit(filename)
