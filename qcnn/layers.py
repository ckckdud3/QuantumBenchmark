from abc import *

import numpy as np
import pennylane as qml


pi = np.pi


# Base class for QCNN components
class QCNNLayer(metaclass=ABCMeta):
    
    def __init__(self, wire_idx: list[int]):

        self.num_wires = len(wire_idx)
        self.wire_idx = wire_idx

        self.num_weights = 0
        self.weight_offset = None
        self.out_idx = None


    def get_wires(self):
        return self.out_idx
    
    
    @abstractmethod
    def __call__(self, w):
        pass



class QCNNEmbedding(QCNNLayer):

    def __init__(self, wire_idx: list[int], depth: int):

        super().__init__(wire_idx)

        self.num_weights = self.num_wires * depth
        self.depth = depth

        self.out_idx = wire_idx


    def __call__(self, data):

        s = self.num_wires

        if len(data.shape) == 1:
            if s == 2:
                for d in range(self.depth):
                    [qml.Hadamard(wires = i) for i in self.wire_idx]
                    [qml.RZ(data[s*d + i], wires = self.wire_idx[i]) for i in range(s)]
                    qml.IsingZZ(0.5*(pi - data[s*d])*(pi - data[s*d+1]), wires = self.wire_idx)
            else:
                for d in range(self.depth):
                    [qml.Hadamard(wires = i) for i in self.wire_idx]
                    [qml.RZ(data[s*d + i], wires = self.wire_idx[i]) for i in range(s)]
                    [qml.IsingZZ(0.5*(pi - data[s*d + i])*(pi - data[s*d + (i + 1)%s]), \
                                wires =[self.wire_idx[i], self.wire_idx[(i + 1)%s]]) for i in range(s)]
        else:
            if s == 2:
                for d in range(self.depth):
                    [qml.Hadamard(wires = i) for i in self.wire_idx]
                    [qml.RZ(data[:,s*d + i], wires = self.wire_idx[i]) for i in range(s)]
                    qml.IsingZZ(0.5*(pi - data[:,s*d])*(pi - data[:,s*d]), wires = self.wire_idx)
            else:
                for d in range(self.depth):
                    [qml.Hadamard(wires = i) for i in self.wire_idx]
                    [qml.RZ(data[:,s*d + i], wires = self.wire_idx[i]) for i in range(s)]
                    [qml.IsingZZ(0.5*(pi - data[:,s*d + i])*(pi - data[:,s*d + (i + 1)%s]), \
                                wires =[self.wire_idx[i], self.wire_idx[(i + 1)%s]]) for i in range(s)]



class QCNNConvolution(QCNNLayer):

    def __init__(self, wire_idx: list[int], depth: int):

        super().__init__(wire_idx)

        self.num_weights = None
        
        if self.num_wires == 2:
            self.num_weights = self.num_wires * depth
        else:
            self.num_weights = 2*self.num_wires * depth
        self.depth = depth

        self.out_idx = wire_idx


    def __call__(self, w):

        s = self.num_wires
        o = self.weight_offset

        if s != 2:
            for d in range(self.depth):
                [qml.Hadamard(wires = i) for i in self.wire_idx]
                for i in range(int(s/2)):
                    qml.CZ(wires = [self.wire_idx[i*2], self.wire_idx[(i*2 + 1)%s]])
                [qml.RX(w[o + s*(2*d) + i], wires = self.wire_idx[i]) for i in range(s)]

                [qml.Hadamard(wires = i) for i in self.wire_idx]
                for i in range(int(s/2)):
                    qml.CZ(wires=[self.wire_idx[i*2 + 1],self.wire_idx[(i*2 + 2)%s]])
                [qml.RX(w[o + s*(2*d+1) + i],wires=self.wire_idx[i]) for i in range(s)]
        else:
            for d in range(self.depth):
                [qml.Hadamard(wires = i) for i in self.wire_idx]
                for i in range(int(s/2)):
                    qml.CZ(wires = [self.wire_idx[i*2], self.wire_idx[(i*2 + 1)%s]])
                [qml.RX(w[o + s*d + i], wires = self.wire_idx[i]) for i in range(s)]



class QCNNPooling(QCNNLayer):

    def __init__(self, measure, target):

        super().__init__([measure, target])

        self.measure_idx = measure
        self.target_idx = target
        self.num_weights = 4

        self.out_idx = [target]


    def __call__(self, w):

        trigger = qml.measure(self.measure_idx)
        o = self.weight_offset

        qml.cond(trigger == 0, qml.RZ)(w[o], wires = self.target_idx)
        qml.cond(trigger == 0, qml.RX)(w[o+1], wires = self.target_idx)

        qml.cond(trigger == 1, qml.RZ)(w[o+2], wires = self.target_idx)
        qml.cond(trigger == 1, qml.RX)(w[o+3], wires = self.target_idx)
