import torch
import pennylane as qml
import numpy as np

import matplotlib.pyplot as plt

from .blocks import *

dev = 'cuda'

class QCNNSequential:

    def __init__(self, block_sequence: list[QCNNBlock], device, num_wires):

        if not isinstance(block_sequence[0], QCNNEmbeddingBlock):
            raise TypeError('First block must be EmbeddingBlock')

        self.sequential = block_sequence
        self.dev = device
        self.out_idx = set([i for i in range(num_wires)])
        self.num_wires = num_wires

        self.embedding_blocks = 0

        for block in block_sequence:
            if isinstance(block, QCNNEmbeddingBlock):
                self.embedding_blocks += 1
            else: break

        tmp = 0
        for block in block_sequence[self.embedding_blocks:]:
            tmp += block.get_num_weights()
            self.out_idx = self.out_idx - (set(block.wire_idx) - set(block.get_wires()))
        
        self.out_idx = list(self.out_idx)
        self.num_eval_weights = 2**len(self.out_idx)
        self.num_integrated_weights = tmp + self.num_eval_weights
        self.eval_offset = tmp

        template = 2*pi*(np.random.rand(self.num_integrated_weights) - 0.5)
        template[-self.num_eval_weights:] = np.random.rand(self.num_eval_weights) - 0.5
        self.w = torch.tensor(template, dtype=torch.float64, requires_grad=True, device=dev)

        self.torch_opt = torch.optim.Adam([self.w], lr=4e-2)
        self.torch_loss = torch.nn.MSELoss()

        tmp = 0
        for block in block_sequence[self.embedding_blocks:]:
            block.integrated_weights_idx = tmp
            block.weight_propagator()
            tmp += block.get_num_weights()
    

    def evaluator(self, probs, w):
        return probs @ w[self.eval_offset:]
    

    def draw_circuit(self, filename):
        
        input_len = 0
        for embed in self.sequential[:self.embedding_blocks]:
            input_len += embed.num_wires * embed.depth
        
        randinput = torch.tensor(np.random.rand(input_len).reshape(1,-1), requires_grad=False).to(dev)
        
        @qml.qnode(self.dev)
        def circuit():
            pivot = 0
            for embed in self.sequential[:self.embedding_blocks]:
                w = embed.num_wires*embed.depth
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
            
        probs = inner_call()
        eval = self.evaluator(probs,self.w)
        return eval
    

    def step(self, data, label):

        def closure():
            labeltensor = torch.tensor(label, dtype=torch.float64).to(dev)
            datatensor = torch.tensor(data, dtype=torch.float64).to(dev)
            self.torch_opt.zero_grad()
            eval = self(datatensor)

            if not torch.is_same_size(eval, labeltensor):
                eval = torch.unsqueeze(eval, -1)

            cost = self.torch_loss(eval, labeltensor)
            cost.backward()
            return cost
        
        self.torch_opt.zero_grad()
        loss = self.torch_opt.step(closure).item()
        return loss
    

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