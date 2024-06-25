from abc import *

from .layers import *
    

class QCNNBlock(metaclass=ABCMeta):

    def __init__(self):
        self.wire_idx = None
        self.integrated_weights_idx = None
        pass


    @abstractmethod
    def __call__(self, w):
        pass


    @abstractmethod
    def get_num_weights(self):
        pass


    @abstractmethod
    def get_wires(self):
        pass


    @abstractmethod
    def weight_propagator(self):
        pass



class QCNNEmbeddingBlock(QCNNBlock):

    def __init__(self, wire_idx: list[int], depth: int):

        super().__init__()
        self.num_wires = len(wire_idx)
        self.depth = depth

        self.wire_idx = wire_idx
        self.out_idx = wire_idx

        self.embed = QCNNEmbedding(self.wire_idx, depth)


    def __call__(self, w):
        self.embed(w)


    def get_num_weights(self):
        return self.embed.num_weights
    

    def get_wires(self):
        return self.out_idx


    def weight_propagator(self):
        pass

class QCNNConvPoolBlock(QCNNBlock):

    def __init__(self, wire_idx: list[int], depth: int):

        assert len(wire_idx) % 2 == 0, 'Number of wires is not even.'

        super().__init__()
        self.num_wires = len(wire_idx)
        self.depth = depth
        self.wire_idx = wire_idx

        self.conv = QCNNConvolution(self.wire_idx, depth)
        self.pool = [QCNNPooling(wire_idx[i], wire_idx[i + 1]) for i in range(0, self.num_wires, 2)]

        self.out_idx = []
        for pool in self.pool:
            self.out_idx += pool.out_idx
        self.layers = [self.conv] + self.pool


    def __call__(self, w):

        self.conv(w)

        for pool in self.pool:
            pool(w)


    def get_num_weights(self):
        return self.conv.num_weights + sum([pool.num_weights for pool in self.pool])
    

    def get_wires(self):
        return self.out_idx
    

    def weight_propagator(self):

        tmp = self.integrated_weights_idx

        self.conv.weight_offset = tmp
        tmp += self.conv.num_weights

        for pool in self.pool:
            pool.weight_offset = tmp
            tmp += pool.num_weights