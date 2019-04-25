import os
import torch.utils.data
import pickle as pkl
import numpy as np


def _normalize_thetao_with_daily_stats(zdata):
    daily_mean = zdata['daily_mean'].reshape(-1, 1, 1)
    daily_std = zdata['daily_std'].reshape(-1, 1, 1)
    zdata['thetao'] = (zdata['thetao'] - daily_mean) / daily_std

def _normalize_thetao(zdata):
    mean = zdata['thetao'].mean(axis=2).mean(axis=1)
    std = zdata['thetao'].reshape(zdata['thetao'].shape[0], -1).std(axis=1)
    zdata['thetao'] = (zdata['thetao'] - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
    #we have to modify mean and std if we renormalize again
    zdata['daily_mean'] = zdata['daily_mean'] + zdata['daily_std'] * mean
    zdata['daily_std'] = zdata['daily_std'] * std

def _rescale_thetao(zdata):
    vmin = zdata['thetao'].min(axis=2).min(axis=1)
    vmax = zdata['thetao'].max(axis=2).max(axis=1)
    rmin = vmin.reshape(-1, 1, 1)
    rmax = vmax.reshape(-1, 1, 1)
    zdata['thetao'] = (zdata['thetao'] - rmin) / (rmax - rmin)
    #we have to modify mean and std if we renormalize again
    zdata['daily_mean'] = zdata['daily_mean'] + zdata['daily_std'] * vmin
    zdata['daily_std'] = zdata['daily_std'] * (vmax - vmin)

def _normalize_uo_vo(zdata):
    # print('nromalizing ')
    norm = (np.abs(zdata['uo']) + np.abs(zdata['vo'])).mean(axis=2).mean(axis=1)
    zdata['uo'] = zdata['uo'] / norm.reshape(-1, 1, 1)
    zdata['vo'] = zdata['vo'] / norm.reshape(-1, 1, 1)
    zdata['uv_norm'] = norm


class SSTSeq(torch.utils.data.Dataset):

    def __init__(self, root, seq_len=4, target_seq_len=6,
                 time_slice=None,
                 normalize_by_day=True,
                 rescale_method=None,
                 transform=None, target_transform=None, 
                 co_transform=None,
                 normalize_uv=True,
                 zones=None):

        if zones is None:  # using all zones
            zones = range(1, 30)

        if time_slice is None:  # using all times
            time_slice = slice(None, None)

        self.root = root
        self.zones = zones
        self.seq_len = seq_len
        self.normalize_by_day = normalize_by_day
        self.rescale_method = rescale_method
        self.target_seq_len = target_seq_len
        self.time_slice = time_slice
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

        self.data = {}
        for zone in zones:
            path = os.path.join(root, 'data_' + str(zone) + '.pkl')
            zdata = pkl.load(open(path, 'rb'))

            if normalize_by_day:
                _normalize_thetao_with_daily_stats(zdata)

            if rescale_method == 'norm':
                print('=> norm rescale zone {}'.format(zone))
                _normalize_thetao(zdata)

            elif rescale_method == 'minmax':
                print('=> minmax rescale zone {}'.format(zone))
                _rescale_thetao(zdata)

            if normalize_uv:
                print('normalizing uv !')
                _normalize_uo_vo(zdata)
            for var in ['thetao', 'uo', 'vo']:
                zdata[var] = zdata[var][time_slice]

            self.data[zone] = zdata
        # print(time_slice)
        # print(f'num days: {len(self.data[self.zones[0]]["thetao"])}')
        self.num_single = self.data[zones[0]]['thetao'].shape[0] - seq_len - target_seq_len + 1
        self.num = self.num_single * len(zones)

        print(f'size: {len(self)} num days: {len(self.data[self.zones[0]]["thetao"])}')

    def __getitem__(self, index):
        zone = self.zones[index // self.num_single]# - 1)]
        # sample_num = index % (self.num_single - 1)
        sample_num = index % self.num_single
        zdata = self.data[zone]

        input = zdata['thetao'][sample_num: sample_num + self.seq_len]
        target = zdata['thetao'][sample_num + self.seq_len: sample_num + self.seq_len + self.target_seq_len]
        uo_target = zdata['uo'][sample_num + self.seq_len: sample_num + self.seq_len + self.target_seq_len]
        vo_target = zdata['vo'][sample_num + self.seq_len: sample_num + self.seq_len + self.target_seq_len]
        w_target = np.concatenate([np.expand_dims(uo_target, 1), np.expand_dims(vo_target, 1)], 1)
        # print('zdata', zdata['uo'].shape, zdata['vo'].shape, 'w_targer', w_target.shape, np.expand_dims(uo_target, 1).shape)
        # exit()
        # 
        return input, target, w_target

    def __len__(self):
        return self.num # - len(self.zones)