import os
import cv2
import glob
import tqdm
import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from models.sunet import SUNet2EnResConv
from utils.meter_loss_functions import balanced_loss_function
from utils.datasets import HDF5DatasetV2
from sklearn.metrics import classification_report
from utils.configs import TrainingConfig



def evaluate_metrics(pred, target, mask=None, deltas=(1.25, 1.25**2, 1.25**3), eps=1e-8):
    """
    Compute depth metrics for a single 2D depth image.

    Args:
        pred   : (H,W) predicted depth (numpy array)
        target : (H,W) ground truth depth (numpy array)
        mask   : (H,W) boolean or 0/1 array (optional).
                 If None, valid = target > 0.
        deltas : thresholds for δ accuracy
        eps    : small constant to avoid division/log issues

    Returns:
        dict with metrics: rmse, rmse_log, abs_rel, sq_rel, δ metrics
    """
    pred = pred[mask].clip(min=eps)
    target = target[mask].clip(min=eps)

    results = {}

    # δ-accuracy
    ratio = np.maximum(pred / target, target / pred)
    for d in deltas:
        results[f"δ<{d:.3f}"] = np.mean(ratio < d)

    diff = pred - target
    log_diff = np.log(pred) - np.log(target)
    
    # RMSE
    results["rmse"] = np.sqrt(np.mean(diff ** 2))

    # RMSE (log)
    results["rmse_log"] = np.sqrt(np.mean(log_diff ** 2))

    # AbsRel
    results["abs_rel"] = np.mean(np.abs(diff) / target)

    # SqRel
    results["sq_rel"] = np.mean((diff ** 2) / target)

    return results


@torch.no_grad()
def evaluate_model(model: nn.Module, metadata_df: pd.DataFrame, hdf5_file: str, device: torch.device, max_depth: float = 200.0):
    model = model.half().eval()
    results = []
    
    df = metadata_df[metadata_df.split == "test"].reset_index(drop=True)
    
    with h5py.File(hdf5_file, "r") as f:
        for i in tqdm.trange(df.shape[0]):
            split, sequence, events_id, prior_depth_frame_id, gt_depth_frame_id, duration = df.iloc[i]

            depth_frame = f[f"{sequence}/depth_frames"][prior_depth_frame_id]
            event_frame = f[f"{sequence}/events"][events_id]
            gt_depth_frame = f[f"{sequence}/depth_frames"][gt_depth_frame_id]
            
            gt_mask = gt_depth_frame < max_depth
            
            gt_depth_frame[gt_depth_frame > max_depth] = 0
            depth_frame[depth_frame > max_depth] = 0

            depth_frame = torch.tensor(depth_frame, dtype=torch.float16, device=device).unsqueeze(0)
            event_frame = torch.tensor(event_frame, dtype=torch.float16, device=device).unsqueeze(0)
            
            pred_depth, _ = model(depth_frame, event_frame)
            pred_depth = pred_depth.detach().squeeze(0).cpu().numpy().astype(np.float32)
            
            metrics_dict = evaluate_metrics(pred_depth, gt_depth_frame, gt_mask)
            
            results.append({
                "split": split,
                "town": sequence.split("_")[0],
                "duration": duration,
                **metrics_dict
            })
            
    return pd.DataFrame(results)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Available device is {device}")
    
    model_dir = "checkpoints/SUNet2EnResConv_20250910_144511"

    metadata_file = "data/extrapolation/variable_interval_voxel_grids_v1_metadata.csv"
    hdf5_file = "data/extrapolation/variable_interval_voxel_grids_v1.h5"

    metadata_df = pd.read_csv(metadata_file)
    metadata_df = metadata_df[metadata_df.split == "test"].reset_index(drop=True)

    # config = TrainingConfig.load(os.path.join(model_dir, "new_config.pkl"))

    model = SUNet2EnResConv(n_channels=16, activation=torch.nn.ReLU()).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device, weights_only=True))

    results = evaluate_model(model, metadata_df, hdf5_file, device, max_depth=200.0)
    
    print(results.groupby("split").mean(numeric_only=True))

if __name__ == "__main__":
    main()