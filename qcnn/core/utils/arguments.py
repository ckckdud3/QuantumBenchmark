from dataclasses import dataclass

@dataclass
class modelarguments:

    scheme:              str
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


@dataclass
class otherarguments:

    dataset_path: str
    save_to:      str