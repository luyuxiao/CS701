import torch
import os


def make_pred_list(num_classes, mask, threshold):
    batch_size = mask.size(0)
    result = torch.zeros(batch_size, num_classes)
    ones = torch.ones(batch_size, num_classes)
    result = mask
    result[mask >= threshold] = 1
    result[mask < threshold] = 0
    return result


def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
