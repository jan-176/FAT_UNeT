import os
import zipfile
import numpy as np
import torch
from torch import nn
import subprocess
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import ndcg_score
from src import models
from src.fat import FAT
from src.unet import UNeT


class LogCoshLoss(torch.nn.Module):
    """
    Logarithm of the hyperbolic cosine of the prediction error.

    Args:
        y_t: torch.Tensor of shape (batch_size, n_features)
            The true values.
        y_prime_t: torch.Tensor of shape (batch_size, n_features)
            The predicted values.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t, y_t):
        ey_t = y_t - y_prime_t
        return torch.log(torch.cosh(ey_t + 1e-12))
    
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def hit_att(ascore, labels, ps = [100, 150]):
    res = {}
    for p in ps:
        hit_score = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0])
            if l:
                size = round(p * len(l) / 100)
                a_p = set(a[:size])
                intersect = a_p.intersection(l)
                hit = len(intersect) / len(l)
                hit_score.append(hit)
        res[f'Hit@{p}%'] = np.mean(hit_score)
    return res

def ndcg(ascore, labels, ps = [100, 150]):
    res = {}
    for p in ps:
        ndcg_scores = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            labs = list(np.where(l == 1)[0])
            if labs:
                k_p = round(p * len(labs) / 100)
                try:
                    hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k = k_p)
                except Exception as e:
                    return {}
                ndcg_scores.append(hit)
        res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
    return res

def execute_command(command):
    try:
        # Run the command and capture the output
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Decode and return the standard output
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        # If the command fails, capture and return the standard error
        return e.stderr.decode('utf-8')

def cut_tensor(percentage: float, tensor: torch.Tensor):
    """
    Cut the tensor by percentage from the middle in the first dimension.

    """
    mid = round(tensor.size(0) / 2)
    window = round(tensor.size(0) * percentage * 0.5)
    return tensor[mid - window: mid + window, :]

def train_val_split(data : torch.Tensor, train_size : float = 0.8, shuffle : bool = False, percentage : float = 1.0):
    """
    Split the data into training and validation sets among the first dimension.
    """

    # set random state for reproducibility
    torch.manual_seed(0)

    if shuffle:
        data = data[torch.randperm(data.size(0))]

    if percentage < 1.0 and percentage > 0.0:
        data = cut_tensor(percentage, data)

    split = int(data.size(0) * train_size)
    return data[:split], data[split:]

def convert_to_windows(data, window_size):
    windows = []
    for i, g in enumerate(data):
        if i >= window_size:
            w = data[(i + 1) - window_size:(i + 1)]  # cut
        else:
            w = torch.cat([data[0].repeat(window_size - (i+1), 1), data[0:(i+1)]])  # pad (added (i+1), because without that, windows[0] and windows[1] are the same)
        windows.append(w)
    return torch.stack(windows)

def load_dataset(device, dataset="", id="", scaler=""):
    dict_train_tensor = {}
    dict_test_tensor = {}
    dict_test_labels = {}

    with zipfile.ZipFile(f'data/{dataset}.zip') as z:
        for file in z.namelist():
            # select only files that contain scaler, and .npy and file.endswith('.npy'):
            if ((scaler in file and dataset == "TIKI") or (id in file and dataset != "TIKI")) and file.endswith('.npy'):
                filename = os.path.basename(file)
                entity, split = filename.split('_')
                
                if "train" in split:
                    dict_train_tensor[entity] = torch.tensor(np.load(z.open(file)),dtype=torch.float64).to(device)
                elif "test" in split:
                    dict_test_tensor[entity] = torch.tensor(np.load(z.open(file)),dtype=torch.float64).to(device)
                elif "labels" in split:
                    dict_test_labels[entity] = np.load(z.open(file))
    
    # mock timestamp from start 1970 in seconds with 30 seconds interval
    start_time = datetime.strptime("2000-06-17 00:00:00", "%Y-%m-%d %H:%M:%S").timestamp()
    dict_test_timestamps = {k: np.arange(start_time, start_time + len(v) * 30, 30) for k, v in dict_test_tensor.items()}

    keys = list(dict_train_tensor.keys())
    _,n_columns = dict_train_tensor[keys[0]].shape
    
    # mack columns called "feature_0", "feature_1", ...
    columns = [f"feature_{i}" for i in range(n_columns)]

    return dict_train_tensor, dict_test_tensor, dict_test_labels, dict_test_timestamps, columns

def load_model(modelname, dims, device, lr_d = 0.0001):
    if ("FAT" in modelname):
        # PARAMETERS
        input_dim = dims
        d_model = 256
        n_heads = 8
        num_encoder_layers = 1
        num_decoder_layers = 1
        ff_hidden_dim = 1024
        
        model = FAT(input_dim, d_model, n_heads, num_encoder_layers, num_decoder_layers, ff_hidden_dim)
    elif ("UNeT" in modelname):
        # PARAMETERS
        input_dim = dims
        d_model = 512
        n_heads = 8
        num_encoder_layers = 1
        num_decoder_layers = 1
        ff_hidden_dim = 1024

        model = UNeT(input_dim, d_model, n_heads, num_encoder_layers, num_decoder_layers, ff_hidden_dim)
    else:
        model_class = getattr(models, modelname)
        model = model_class(dims)
        
    model.double().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_d, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    epoch = -1
    return model, optimizer, scheduler, epoch


