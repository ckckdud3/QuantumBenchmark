from abc import *

from .layers import *
    

class QCNNBlock(metaclass=ABCMeta):

    def __init__(self):
        self.wire_idx = None
        self.integrated_weights_idx = None
        self.embedding = False
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



class ZZFeatureMapEmbeddingBlock(QCNNBlock):

    def __init__(self, wire_idx: list[int], depth: int):

        super().__init__()
        self.num_wires = len(wire_idx)
        self.depth = depth

        self.wire_idx = wire_idx
        self.out_idx = wire_idx

        self.embed = QCNNEmbedding(self.wire_idx, depth)

        self.embedding = True


    def __call__(self, w):

        s = self.num_wires

        if s == 2:
            for d in range(self.depth):
                [qml.Hadamard(wires = i) for i in self.wire_idx]
                [qml.RZ(w[:,s*d + i], wires = self.wire_idx[i]) for i in range(s)]
                qml.IsingZZ(0.5*(pi - w[:,s*d])*(pi - w[:,s*d]), wires = self.wire_idx)
        else:
            for d in range(self.depth):
                [qml.Hadamard(wires = i) for i in self.wire_idx]
                [qml.RZ(w[:,s*d + i], wires = self.wire_idx[i]) for i in range(s)]
                [qml.IsingZZ(0.5*(pi - w[:,s*d + i])*(pi - w[:,s*d + (i + 1)%s]), \
                            wires =[self.wire_idx[i], self.wire_idx[(i + 1)%s]]) for i in range(s)]


    def get_num_weights(self):
        return self.embed.num_weights
    

    def get_wires(self):
        return self.out_idx


    def weight_propagator(self):
        pass



class AngleEmbeddingBlock(QCNNBlock):

    def __init__(self, wire_idx: list[int], depth: int):

        super().__init__()

        self.num_wires = len(wire_idx)
        self.num_weights = len(wire_idx)
        self.depth = depth

        self.wire_idx = wire_idx
        self.out_idx = wire_idx

        self.embedding = True


    def __call__(self, w):
        [qml.Hadamard(wires=wire) for wire in self.wire_idx]
        [qml.RZ(w[:,i], wires=wire) for i,wire in enumerate(self.wire_idx)]


    def get_num_weights(self):
        return self.num_weights
    

    def get_wires(self):
        return self.out_idx


    def weight_propagator(self):
        pass



class AmplitudeEmbeddingBlock(QCNNBlock):
        
    def __init__(self, wire_idx: list[int]):

        super().__init__()

        self.num_wires = len(wire_idx)
        self.num_weights = 2**len(wire_idx)

        self.wire_idx = wire_idx
        self.out_idx = wire_idx

        self.embedding = True


    def __call__(self, w):
        w = torch.nn.functional.normalize(w, dim=1)
        qml.StatePrep(w, wires=self.wire_idx)


    def get_num_weights(self):
        return self.num_weights
    

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



class QCNNCrossConvBlock(QCNNBlock):

    def __init__(self,upper_wire_idx: list[int], lower_wire_idx: list[int], depth: int):
        
        assert len(upper_wire_idx) == len(lower_wire_idx), 'Number of wires not matching'

        self.w_per_block = len(upper_wire_idx)
        
        # assert self.w_per_block % 2 == 0, 'Number of wires per block is not even'

        super().__init__()

        self.upper_wire_idx = upper_wire_idx
        self.lower_wire_idx = lower_wire_idx
        self.wire_idx = upper_wire_idx + lower_wire_idx
        
        self.num_wires = len(upper_wire_idx) + len(lower_wire_idx)

        self.upper_pool_measure = self.upper_wire_idx[::2]
        self.lower_pool_measure = self.lower_wire_idx[::2]
        self.upper_pool_out = self.upper_wire_idx[1::2]
        self.lower_pool_out = self.lower_wire_idx[1::2]

        if self.w_per_block % 2:
            self.upper_pool_measure = self.upper_pool_measure[:-1]
            self.lower_pool_measure = self.lower_pool_measure[:-1]

        upper_out = set(upper_wire_idx) - set(self.upper_pool_measure)
        lower_out = set(lower_wire_idx) - set(self.lower_pool_measure)

        self.out_idx = list(upper_out | lower_out)

        self.upper_conv = QCNNConvolution(self.upper_wire_idx, depth)
        self.lower_conv = QCNNConvolution(self.lower_wire_idx, depth)

        measure_target_tuples_upper = []
        measure_target_tuples_lower = []
        
        for i in range(len(self.upper_pool_out)):
            measure_target_tuples_upper.append((self.upper_pool_measure[i], self.lower_pool_out[i]))
            measure_target_tuples_lower.append((self.lower_pool_measure[i], self.upper_pool_out[i]))

        self.cross_pool = [QCNNPooling(p,t) for (p,t) in measure_target_tuples_upper] \
                        + [QCNNPooling(p,t) for (p,t) in measure_target_tuples_lower]
        
        self.layers = [self.upper_conv, self.lower_conv] + self.cross_pool


    def __call__(self, w):

        self.upper_conv(w)
        self.lower_conv(w)

        for pool in self.cross_pool:
            pool(w)


    def get_num_weights(self):
        return self.upper_conv.num_weights + self.lower_conv.num_weights + sum([pool.num_weights for pool in self.cross_pool])
    

    def get_wires(self):
        return self.out_idx
    

    def weight_propagator(self):
        
        tmp = self.integrated_weights_idx

        self.upper_conv.weight_offset = tmp
        tmp += self.upper_conv.num_weights

        self.lower_conv.weight_offset = tmp
        tmp += self.lower_conv.num_weights

        for pool in self.cross_pool:

            pool.weight_offset = tmp
            tmp += pool.num_weights



class QCNNCyclicCrossConvBlock(QCNNBlock):
    
    def __init__(self, wire_groups: list[list[int]], depth: int):
        
        super().__init__()
        test_len = len(wire_groups[0])
        self.wire_idx = []
        for wires in wire_groups:
            assert len(wires) == test_len, 'Number of wires not matching'
            self.wire_idx += wires

        self.w_per_block = test_len
        
        assert self.w_per_block % 2 == 0, 'Number of wires per block is not even'

        self.source_indices = [wires[0::2] for wires in wire_groups]
        self.target_indices = [wires[1::2] for wires in wire_groups]

        self.out_idx = []
        for t in self.target_indices:
            self.out_idx += t
        self.convs = [QCNNConvolution(w, depth) for w in wire_groups]

        self.cyclic_pools: list[QCNNPooling] = []

        l = len(self.source_indices)
        for i in range(l):
            for wire_s, wire_t in zip(self.source_indices[i], self.target_indices[(i+1)%l]):
                self.cyclic_pools.append(QCNNPooling(wire_s, wire_t))
        
        self.layers = self.convs + self.cyclic_pools


    def __call__(self, w):

        for conv in self.convs:
            conv(w)
        for pool in self.cyclic_pools:
            pool(w)


    def get_num_weights(self):
        return sum([conv.num_weights for conv in self.convs]) + 4*len(self.cyclic_pools)
    

    def get_wires(self):
        return self.out_idx
    

    def weight_propagator(self):
        
        tmp = self.integrated_weights_idx
        
        for conv in self.convs:
            conv.weight_offset = tmp
            tmp += conv.num_weights

        for pool in self.cyclic_pools:
            pool.weight_offset = tmp
            tmp += pool.num_weights



class QCNNTwoLocalBlock(QCNNBlock):
    
    def __init__(self, wire_idx: list[int], depth: int):

        assert len(wire_idx) % 2 == 0, 'Number of wires is not even.'

        super().__init__()
        self.num_wires = len(wire_idx)
        self.depth = depth
        self.wire_idx = wire_idx

        self.block = TwoLocalAnsatz(self.wire_idx, depth)
        self.pool = [QCNNPooling(wire_idx[i], wire_idx[i + 1]) for i in range(0, self.num_wires, 2)]

        self.out_idx = []
        for pool in self.pool:
            self.out_idx += pool.out_idx


    def __call__(self, w):

        self.block(w)

        for pool in self.pool:
            pool(w)


    def get_num_weights(self):
        return self.block.num_weights + sum([pool.num_weights for pool in self.pool])
    

    def get_wires(self):
        return self.out_idx
    

    def weight_propagator(self):

        tmp = self.integrated_weights_idx

        self.block.weight_offset = tmp
        tmp += self.block.num_weights

        for pool in self.pool:
            pool.weight_offset = tmp
            tmp += pool.num_weights


class QCNNCrossTwoLocalBlock(QCNNBlock):

    def __init__(self,upper_wire_idx: list[int], lower_wire_idx: list[int], depth: int):
        
        assert len(upper_wire_idx) == len(lower_wire_idx), 'Number of wires not matching'

        self.w_per_block = len(upper_wire_idx)
        
        # assert self.w_per_block % 2 == 0, 'Number of wires per block is not even'

        super().__init__()

        self.upper_wire_idx = upper_wire_idx
        self.lower_wire_idx = lower_wire_idx
        self.wire_idx = upper_wire_idx + lower_wire_idx
        
        self.num_wires = len(upper_wire_idx) + len(lower_wire_idx)

        self.upper_pool_measure = self.upper_wire_idx[::2]
        self.lower_pool_measure = self.lower_wire_idx[::2]
        self.upper_pool_out = self.upper_wire_idx[1::2]
        self.lower_pool_out = self.lower_wire_idx[1::2]

        if self.w_per_block % 2:
            self.upper_pool_measure = self.upper_pool_measure[:-1]
            self.lower_pool_measure = self.lower_pool_measure[:-1]

        upper_out = set(upper_wire_idx) - set(self.upper_pool_measure)
        lower_out = set(lower_wire_idx) - set(self.lower_pool_measure)

        self.out_idx = list(upper_out | lower_out)

        self.upper_block = TwoLocalAnsatz(self.upper_wire_idx, depth)
        self.lower_block = TwoLocalAnsatz(self.lower_wire_idx, depth)

        measure_target_tuples_upper = []
        measure_target_tuples_lower = []
        
        for i in range(len(self.upper_pool_out)):
            measure_target_tuples_upper.append((self.upper_pool_measure[i], self.lower_pool_out[i]))
            measure_target_tuples_lower.append((self.lower_pool_measure[i], self.upper_pool_out[i]))

        self.cross_pool = [QCNNPooling(p,t) for (p,t) in measure_target_tuples_upper] \
                        + [QCNNPooling(p,t) for (p,t) in measure_target_tuples_lower]
        
        self.layers = [self.upper_block, self.lower_block] + self.cross_pool


    def __call__(self, w):

        self.upper_block(w)
        self.lower_block(w)

        for pool in self.cross_pool:
            pool(w)


    def get_num_weights(self):
        return self.upper_block.num_weights + self.lower_block.num_weights + sum([pool.num_weights for pool in self.cross_pool])
    

    def get_wires(self):
        return self.out_idx
    

    def weight_propagator(self):
        
        tmp = self.integrated_weights_idx

        self.upper_block.weight_offset = tmp
        tmp += self.upper_block.num_weights

        self.lower_block.weight_offset = tmp
        tmp += self.lower_block.num_weights

        for pool in self.cross_pool:

            pool.weight_offset = tmp
            tmp += pool.num_weights



class QCNNCyclicTwoLocalBlock(QCNNBlock):

    def __init__(self, wire_groups: list[list[int]], depth: int):
        
        super().__init__()
        test_len = len(wire_groups[0])
        self.wire_idx = []
        for wires in wire_groups:
            assert len(wires) == test_len, 'Number of wires not matching'
            self.wire_idx += wires

        self.w_per_block = test_len
        
        assert self.w_per_block % 2 == 0, 'Number of wires per block is not even'

        self.source_indices = [wires[0::2] for wires in wire_groups]
        self.target_indices = [wires[1::2] for wires in wire_groups]

        self.out_idx = []
        for t in self.target_indices:
            self.out_idx += t
        self.convs = [TwoLocalAnsatz(w, depth) for w in wire_groups]

        self.cyclic_pools: list[QCNNPooling] = []

        l = len(self.source_indices)
        for i in range(l):
            for wire_s, wire_t in zip(self.source_indices[i], self.target_indices[(i+1)%l]):
                self.cyclic_pools.append(QCNNPooling(wire_s, wire_t))
        
        self.layers = self.convs + self.cyclic_pools


    def __call__(self, w):

        for conv in self.convs:
            conv(w)
        for pool in self.cyclic_pools:
            pool(w)


    def get_num_weights(self):
        return sum([conv.num_weights for conv in self.convs]) + 4*len(self.cyclic_pools)
    

    def get_wires(self):
        return self.out_idx
    

    def weight_propagator(self):
        
        tmp = self.integrated_weights_idx
        
        for conv in self.convs:
            conv.weight_offset = tmp
            tmp += conv.num_weights

        for pool in self.cyclic_pools:
            pool.weight_offset = tmp
            tmp += pool.num_weights