from datasets import load_dataset
import argparse
import json
import time
import torch
from torch.utils.data import DistributedSampler, DataLoader

from meldataset import MelDataset, get_dataset_filelist
from env import AttrDict, build_env
from models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
    Encoder,
    Quantizer,
)


parser = argparse.ArgumentParser()

parser.add_argument("--group_name", default=None)
parser.add_argument("--input_wavs_dir", default="../datasets/audios")
parser.add_argument("--input_mels_dir", default=None)
parser.add_argument("--input_training_file", default="../datasets/training.txt")
parser.add_argument("--input_validation_file", default="../datasets/validation.txt")
parser.add_argument("--checkpoint_path", default="checkpoints")
parser.add_argument("--config", default="config.json")
parser.add_argument("--training_epochs", default=200, type=int)
parser.add_argument("--stdout_interval", default=5, type=int)
parser.add_argument("--checkpoint_interval", default=5000, type=int)
parser.add_argument("--summary_interval", default=100, type=int)
parser.add_argument("--validation_interval", default=10000, type=int)
parser.add_argument("--fine_tuning", default=False, type=bool)

a = parser.parse_args()

training_filelist, validation_filelist = get_dataset_filelist(a)

with open(a.config) as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)
# print(h)
rank = 0
device = torch.device("cuda:{:d}".format(rank))
print(device)
gs = load_dataset(
    "speechcolab/gigaspeech", "xs", use_auth_token=True, trust_remote_code=True
)

encoder = Encoder(h).to(device)
quantizer = Quantizer(h).to(device)
for i, item in enumerate(gs["train"]):
    y = item["audio"]["array"]
    y = torch.tensor(y, dtype=torch.float32).to(device)
    y = y.unsqueeze(0).unsqueeze(0)
    print("y", y.shape)
    c = encoder(y)
    print("c", c.shape)
    q, loss_q, c = quantizer(c)
    print("q", q.shape)
    print("loss_q", loss_q)
    print("c", len(c))
    print("c", c[0].shape)
    break

# trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
#                           h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
#                           shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
#                           fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)
