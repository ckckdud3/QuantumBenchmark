import torch
import pennylane as qml
import numpy as np

import matplotlib.pyplot as plt

from .blocks import *
from .utils.arguments import modelarguments

dev = 'cuda:0'

pi = np.pi

class QCNNSequential:

    def __init__(self, block_sequence: list[QCNNBlock], device, args:modelarguments):

        if not block_sequence[0].embedding:
            raise TypeError('First block must be EmbeddingBlock')

        self.sequential = block_sequence
        self.args = args
        self.dev = device
        self.out_idx = set([i for i in range(args.num_wires)])
        self.num_wires = args.num_wires
        self.out_dim = args.out_dim

        self.embedding_blocks = 0

        for block in block_sequence:
            if block.embedding:
                self.embedding_blocks += 1
            else: break

        tmp = 0
        for block in block_sequence[self.embedding_blocks:]:
            tmp += block.get_num_weights()
            self.out_idx = self.out_idx - (set(block.wire_idx) - set(block.get_wires()))
        
        self.out_idx = list(self.out_idx)
        self.num_eval_weights = 2**len(self.out_idx)
        self.num_integrated_weights = tmp + self.num_eval_weights*self.out_dim
        self.eval_offset = tmp

        template = 2*pi*(np.random.rand(self.num_integrated_weights) - 0.5)
        template[-self.num_eval_weights:] = np.random.rand(self.num_eval_weights) - 0.5
        self.w = torch.tensor(template, dtype=torch.float64, requires_grad=True, device=dev)


        self.torch_opt = torch.optim.Adam([self.w], lr=3e-2)
        self.torch_loss = torch.nn.CrossEntropyLoss()

        tmp = 0
        for block in block_sequence[self.embedding_blocks:]:
            block.integrated_weights_idx = tmp
            block.weight_propagator()
            tmp += block.get_num_weights()
    

    def evaluator(self, probs, w):
        return probs @ w[self.eval_offset:].reshape(-1,self.out_dim)
    

    def draw_circuit(self, filename='test'):
        
        input_len = 0
        for embed in self.sequential[:self.embedding_blocks]:
            input_len += embed.get_num_weights()

        randinput = torch.tensor(np.random.rand(input_len).reshape(1,-1), requires_grad=False).to(dev)
        
        @qml.qnode(self.dev)
        def circuit():
            pivot = 0
            for embed in self.sequential[:self.embedding_blocks]:
                w = embed.get_num_weights()
                embed(randinput[:,pivot:pivot+w])
                pivot += w

            for block in self.sequential[self.embedding_blocks:]:
                qml.Barrier(wires=range(self.num_wires), only_visual=True)
                block(self.w)

            return qml.probs(wires=self.out_idx)
        
        fig, ax = qml.draw_mpl(circuit, style='pennylane')()
        fig.savefig(f'{filename}.png')
        

    def get_weights(self):
        return self.w
    

    def update_weights(self, w):
        self.w = w


    def __call__(self, data):
        
        @qml.qnode(self.dev, interface='torch')
        def inner_call():
            
            pivot = 0
            for embedding in self.sequential[:self.embedding_blocks]:

                num_w = embedding.get_num_weights()
                embedding(data[:,pivot:pivot+num_w])
                pivot += num_w


            for block in self.sequential[self.embedding_blocks:]:
                block(self.w)
            return qml.probs(wires=self.out_idx)

        if self.args.embed_type == 'amp' and self.args.num_processor > 1:
            tr = qml.transforms.broadcast_expand(inner_call)
            probs = tr()
            eval = self.evaluator(probs,self.w)
            return eval

        probs = inner_call()
        eval = self.evaluator(probs,self.w)
        return eval
    

    def step(self, data, label):
        
        self.torch_opt.zero_grad()
        eval = self(data)

        if not torch.is_same_size(eval, label):
            eval = torch.unsqueeze(eval, -1)

        cost = self.torch_loss(eval, label.reshape(-1,1))
        cost.backward()
        self.torch_opt.step()

        var = torch.var(self.w.grad).item()
        norm = torch.norm(self.w.grad, 2.0).item()
        return cost.item(), var, norm
    

    def init_params(self):
        
        template = 2*pi*np.random.rand(self.num_integrated_weights)
        template[-self.num_eval_weights:] = np.random.rand(self.num_eval_weights) - 0.5
        self.w = torch.tensor(template, dtype=torch.float64, requires_grad=True, device=dev)

    
    def fim(self):

        input_len = 0
        for embed in self.sequential[:self.embedding_blocks]:
            input_len += embed.num_wires * embed.depth
        
        randinput = torch.tensor(2*pi*np.random.rand(input_len), requires_grad=False).to(dev)

        @qml.qnode(self.dev, interface='torch')
        def inner_call(w):
            pivot = 0
            for embed in self.sequential[:self.embedding_blocks]:
                weights = embed.num_wires*embed.depth
                embed(randinput[pivot:pivot+weights])
                pivot += weights
                
            for block in self.sequential[self.embedding_blocks:]:
                block(w)
            return qml.probs(wires=self.out_idx)
            
        fim = qml.qinfo.classical_fisher(inner_call)(self.w[:-self.num_eval_weights]).detach().cpu().numpy()
        return fim


def scheme_builder(arg: modelarguments, data_dim):

    scheme_list = ['cyclic', 'cross', 'no_comm', 'full']
    embed_type = ['amp', 'angle', 'zz']
    block_type = ['conv', 'twolocal']

    assert arg.scheme in scheme_list, 'Invalid scheme'
    assert arg.block_type in block_type, 'Invalid block type'
    assert arg.embed_type in embed_type, 'Invalid embed type'

    if arg.scheme == 'full':
        arg.num_processor = 1
    else:
        assert arg.num_processor != 1, 'Invalid number of processor'


    num_block = np.log2(arg.num_wires/arg.num_obs)

    assert num_block == int(num_block), \
        f'Expected np.log2({arg.num_wires}/{arg.num_obs}) to be integer, got {num_block}'
    
    num_block = int(num_block)


    assert arg.num_wires % arg.num_processor == 0, 'Number of wire is not divisible by Number of processor'
    
    wire_per_processor = int(arg.num_wires / arg.num_processor)


    sliced_list = [list(range(i,i+wire_per_processor)) for i in range(0, arg.num_wires, wire_per_processor)]

    block_list = []

    if arg.embed_type == 'angle':

        block = AngleEmbeddingBlock

        embed_per_p = data_dim/(arg.num_wires)
        
        assert embed_per_p == int(embed_per_p), 'invalid data dimension'

        for _ in range(int(embed_per_p)):
            for l in sliced_list:
                block_list.append(block(l, 1))
    
    elif arg.embed_type == 'zz':

        block = ZZFeatureMapEmbeddingBlock

        embed_per_p = data_dim/(arg.num_wires)
        
        assert embed_per_p == int(embed_per_p), 'invalid data dimension'

        for _ in range(int(embed_per_p)):
            for l in sliced_list:
                block_list.append(block(l, 1))

    else:

        block = AmplitudeEmbeddingBlock

        assert arg.num_processor*(2**wire_per_processor) == data_dim, 'invalid data dimension'

        for l in sliced_list:
            block_list.append(block(l))

    if arg.block_type == 'conv':

        if arg.scheme == 'no_comm':

            block = QCNNConvPoolBlock
            
            for i in range(num_block):
                for j in range(len(sliced_list)):
                    block_list.append(block(sliced_list[j], arg.depth))
                for j in range(len(sliced_list)):
                    sliced_list[j] = sliced_list[j][1::2]

        elif arg.scheme == 'cyclic':

            block = QCNNCyclicCrossConvBlock

            for i in range(num_block):
                block_list.append(block(sliced_list, arg.depth))
                for j in range(len(sliced_list)):
                    sliced_list[j] = sliced_list[j][1::2]

        elif arg.scheme == 'cross':

            block = QCNNCrossConvBlock

            for i in range(num_block):
                for j in range(0, len(sliced_list), 2):
                    block_list.append(block(sliced_list[j], sliced_list[j+1], arg.depth))
                for j in range(len(sliced_list)):
                    sliced_list[j] = sliced_list[j][1::2]
                    
        else:
            
            block = QCNNConvPoolBlock

            for i in range(num_block):
                block_list.append(block(sliced_list[0], arg.depth))
                for j in range(len(sliced_list)):
                    sliced_list[j] = sliced_list[j][1::2]

    else:

        if arg.scheme == 'no_comm':

            block = QCNNTwoLocalBlock
            
            for i in range(num_block):
                for j in range(len(sliced_list)):
                    block_list.append(block(sliced_list[j], arg.depth))
                for j in range(len(sliced_list)):
                    sliced_list[j] = sliced_list[j][1::2]

        elif arg.scheme == 'cyclic':

            block = QCNNCyclicTwoLocalBlock

            for i in range(num_block):
                block_list.append(block(sliced_list, arg.depth))
                for j in range(len(sliced_list)):
                    sliced_list[j] = sliced_list[j][1::2]

        elif arg.scheme == 'cross':

            block = QCNNCrossTwoLocalBlock

            for i in range(num_block):
                for j in range(0, len(sliced_list), 2):
                    block_list.append(block(sliced_list[j], sliced_list[j+1], arg.depth))
                for j in range(len(sliced_list)):
                    sliced_list[j] = sliced_list[j][1::2]
                    
        else:
            
            block = QCNNTwoLocalBlock

            for i in range(num_block):
                block_list.append(block(sliced_list[0], arg.depth))
                for j in range(len(sliced_list)):
                    sliced_list[j] = sliced_list[j][1::2]

    
    return block_list