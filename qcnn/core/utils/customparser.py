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
                               num_wires           = self.arg['num_wires'], \
                               depth               = self.arg['depth'], \
                               num_obs             = self.arg['num_obs'], \
                               num_processor       = self.arg['num_processor']),

                trainarguments(batch_size        = self.arg['batch_size'], \
                               num_epoch         = self.arg['num_epoch'],
                               shuffle_per_epoch = self.arg['shuffle_per_epoch']),

                otherarguments(dataset_path = self.arg['dataset_path'], \
                               save_to      = self.arg['save_to']))