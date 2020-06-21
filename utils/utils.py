import h5py
import torch
import shutil
import os
import numpy as np

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, checkpoint_save_dir, dataset, epoch, test_mae, val_mae):
    checkpoint_filepath = os.path.join(checkpoint_save_dir, 'Dataset_' + dataset + str(epoch) + '_' + str(test_mae) + '_' + str(val_mae) + 'checkpoint.pth.tar')
    # checkpoint_filepath = '/home/rainkeeper/Projects/PycharmProjects/CSRNet/checkpoints/' + 'Dataset_' + dataset + str(epoch) + 'checkpoint.pth.tar'
    torch.save(state, checkpoint_filepath)
