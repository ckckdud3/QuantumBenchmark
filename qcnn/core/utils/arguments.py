from dataclasses import dataclass

@dataclass
class modelarguments:

    scheme:              str
    embed_type:          str
    block_type:          str
    num_wires:           int
    depth:               int
    num_obs:             int
    num_processor:       int
    out_dim:             int

@dataclass
class trainarguments:

    batch_size:        int
    num_epoch:         int
    shuffle_per_epoch: bool
    data_dim:          int

@dataclass
class otherarguments:

    dataset_path: str
    save_to:      str