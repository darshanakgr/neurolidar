import os
import cv2
import h5py
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

        
class DepthExtDataset(Dataset):
    def __init__(self, hdf5_file: str, df: pd.DataFrame, clip: bool = False, downsample: float = 0.0):
        super().__init__()
        self.data = h5py.File(hdf5_file, "r")
        self.df = df
        self.clip = clip
        self.downsample = downsample

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        _, sequence, events_id, prior_depth_frame_id, gt_depth_frame_id, _ = self.df.iloc[idx]
        
        event_frame = self.data[f"{sequence}/events"][events_id]
        prior_depth_frame = self.data[f"{sequence}/depth_frames"][prior_depth_frame_id]
        gt_depth_frame = self.data[f"{sequence}/depth_frames"][gt_depth_frame_id]
        
        if event_frame.ndim < 2:
            event_frame = np.reshape(event_frame, (-1, 480, 640))
            
        if self.downsample > 0.0:
            prior_depth_frame = cv2.resize(prior_depth_frame.squeeze(), dsize=None, fx=self.downsample, fy=self.downsample, interpolation=cv2.INTER_AREA)
            gt_depth_frame = cv2.resize(gt_depth_frame.squeeze(), dsize=None, fx=self.downsample, fy=self.downsample, interpolation=cv2.INTER_AREA)
            
            prior_depth_frame = np.expand_dims(prior_depth_frame, axis=0)
            gt_depth_frame = np.expand_dims(gt_depth_frame, axis=0)
            
            new_event_frames = []
            for i in range(event_frame.shape[0]):
                ef = cv2.resize(event_frame[i], dsize=None, fx=self.downsample, fy=self.downsample, interpolation=cv2.INTER_NEAREST)
                new_event_frames.append(ef)
                
            event_frame = np.stack(new_event_frames, axis=0)

        if self.clip:
            prior_depth_frame = np.clip(prior_depth_frame, 0, 200)
            gt_depth_frame = np.clip(gt_depth_frame, 0, 200)
        
        return prior_depth_frame, event_frame, gt_depth_frame
    
    def close(self):
        self.data.close()
         
        
class KeyFrameDataset(Dataset):
    def __init__(self, hdf5_file, split="train"):
        self.hdf5_file = hdf5_file
        self.split = split
        self.data = h5py.File(self.hdf5_file, "r")
        
    def __len__(self):
        if self.split == "train":
            return self.data["train"]["event_frames"].shape[0]
        elif self.split == "test":
            return self.data["test"]["event_frames"].shape[0]
        else:
            raise ValueError("Invalid split. Choose 'train' or 'test'.")
    
    def __getitem__(self, idx):
        event_frames = self.data[self.split]["event_frames"][idx]
        scores = self.data[self.split]["scores"][idx]
        
        if event_frames.ndim == 2:
            event_frames = np.expand_dims(event_frames, axis=0)
        
        scores = np.expand_dims(scores, axis=0)  # Add channel dimension
        event_frames = torch.tensor(event_frames, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)
        return event_frames, scores
    
    def close(self):
        self.data.close()
