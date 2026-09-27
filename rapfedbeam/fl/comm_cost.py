"""
fl/comm_cost.py
Responsibility: count bytes exchanged per FL round for the bytes-vs-utility comparison (E4).
Only backbone parameters count -- adapters and heads never leave the node (per Sec. 4 of plan).
"""


def count_backbone_params(backbone_state_dict):
    # PSEUDOCODE:
    # total = sum(tensor.numel() for tensor in backbone_state_dict.values())
    # return total   # number of scalar parameters
    return sum(int(tensor.numel()) for tensor in backbone_state_dict.values())


def bytes_per_round(mode, num_participating_nodes, num_backbone_params, bytes_per_float=4):
    # PSEUDOCODE:
    # local_only:       0 bytes (no communication at all)
    # fedavg:            each participating node sends delta_theta (upload) AND receives
    #                     the averaged update back (download)
    #                     total = num_participating_nodes * num_backbone_params * bytes_per_float * 2
    # topology_kernel:   same upload cost as fedavg, same download cost
    #                     (personalization changes HOW updates are combined, not how much
    #                      raw data moves per node -- so cost per round is comparable to FedAvg;
    #                      the win in E4 comes from needing FEWER rounds / less total, not
    #                      cheaper individual rounds)
    # return total_bytes_this_round
    if mode == "local_only":
        return 0
    return int(num_participating_nodes * num_backbone_params * bytes_per_float * 2)


def cumulative_bytes(bytes_per_round_list):
    # PSEUDOCODE: running sum, used for the utility-vs-cumulative-bytes plot
    total = 0
    result = []
    for value in bytes_per_round_list:
        total += value
        result.append(total)
    return result
