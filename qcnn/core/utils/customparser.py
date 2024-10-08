from .arguments import *
import yaml

class customparser:

    def __init__(self, file_path):
        self.arg = None
        with open(file_path) as f:
            self.arg = yaml.safe_load(f)
            f.close()

    def parse_custom_args(self):

        return (modelarguments(scheme              = self.arg['scheme'], \
                               embed_type          = self.arg['embed_type'], \
                               block_type          = self.arg['block_type'], \
                               num_wires           = self.arg['num_wires'], \
                               depth               = self.arg['depth'], \
                               num_obs             = self.arg['num_obs'], \
                               num_processor       = self.arg['num_processor'], \
                               out_dim             = self.arg['out_dim']),

                trainarguments(batch_size        = self.arg['batch_size'], \
                               num_epoch         = self.arg['num_epoch'],
                               shuffle_per_epoch = self.arg['shuffle_per_epoch'], \
                               data_dim          = self.arg['data_dim']),

                otherarguments(dataset_path = self.arg['dataset_path'], \
                               save_to      = self.arg['save_to']))