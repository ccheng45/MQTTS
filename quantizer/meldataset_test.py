import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import json
import time
import torch
from torch.utils.data import DistributedSampler, DataLoader

from meldataset import MelDataset, get_dataset_filelist
from env import AttrDict, build_env

parser = argparse.ArgumentParser()

parser.add_argument('--group_name', default=None)
parser.add_argument('--input_wavs_dir', default='../datasets/audios')
parser.add_argument('--input_mels_dir', default=None)
parser.add_argument('--input_training_file', default='../datasets/training.txt')
parser.add_argument('--input_validation_file', default='../datasets/validation.txt')
parser.add_argument('--checkpoint_path', default='checkpoints')
parser.add_argument('--config', default='config.json')
parser.add_argument('--training_epochs', default=200, type=int)
parser.add_argument('--stdout_interval', default=5, type=int)
parser.add_argument('--checkpoint_interval', default=5000, type=int)
parser.add_argument('--summary_interval', default=100, type=int)
parser.add_argument('--validation_interval', default=10000, type=int)
parser.add_argument('--fine_tuning', default=False, type=bool)

a = parser.parse_args()

training_filelist, validation_filelist = get_dataset_filelist(a)

with open(a.config) as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)
print(h)
rank =0
device = torch.device('cuda:{:d}'.format(rank))
print(device)

trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)
print(len(trainset))
# train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                            sampler=None,
                            batch_size=h.batch_size,
                            pin_memory=True,
                            drop_last=True)

for i, batch in enumerate(train_loader):
        # (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze(), speaker_embedding)
        x, y, _, y_mel, spkr = batch
        print(i)
        print("x", x.shape)
        print("y", y.shape)
        print("y_mel", y_mel.shape)
        print(torch.isclose(x, y_mel, rtol=1e-05, atol=1e-08).all())
        print("spkr", spkr.shape)
        # x = torch.autograd.Variable(x.to(device, non_blocking=True))
        # y = torch.autograd.Variable(y.to(device, non_blocking=True))
        # y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
        # y = y.unsqueeze(1)
        break
