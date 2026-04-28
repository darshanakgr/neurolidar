import os
import cv2
import h5py
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

# class CarlaSimDataset(Dataset):
#     def __init__(self, sequence_dirs, delta):
#         self.samples = []
        
#         for sequence_dir in sequence_dirs:
#             timestamps = np.array([int(fname.split(".")[0]) for fname in os.listdir(os.path.join(sequence_dir, "depth"))])
#             timestamps.sort()
            
#             for current_t in timestamps:
#                 next_t = timestamps[np.searchsorted(timestamps, current_t + delta) - 1]

#                 if current_t == next_t:
#                     continue
                
#                 indices = np.argwhere(np.logical_and(timestamps >= current_t, timestamps <= next_t))
#                 event_ts = timestamps[indices].flatten()

#                 self.samples.append((sequence_dir, current_t, next_t, event_ts))

            
#     def _decode_depth_image(self, depth_image):
#         depth_image = depth_image.astype(np.float32)
#         depth_image = ((depth_image[:, :, 2] + depth_image[:, :, 1] * 256 + depth_image[:, :, 0] * (256 ** 2)) / (256 ** 3 - 1))
#         depth_image *= 1000
        
#         return depth_image
    
#     def _create_event_frame(self, image_size, events):
#         xs = events["x"]
#         ys = events["y"]
#         pols = events["pol"]
#         img = np.zeros((*image_size, 3), dtype=np.uint8)
#         img[ys, xs, 2 * pols] = 255
#         return img
            
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         sequence_dir, current_t, next_t, event_ts = self.samples[idx]
        
#         current_depth_img = cv2.imread(os.path.join(sequence_dir, "depth", f"{current_t}.png"))
#         next_depth_img = cv2.imread(os.path.join(sequence_dir, "depth", f"{next_t}.png"))

#         current_mask = cv2.imread(os.path.join(sequence_dir, "mask", f"{current_t}.png"), cv2.IMREAD_UNCHANGED)
#         next_mask = cv2.imread(os.path.join(sequence_dir, "mask", f"{next_t}.png"), cv2.IMREAD_UNCHANGED)

#         current_mask = current_mask / 255
#         next_mask = next_mask / 255

#         current_depth_img = self._decode_depth_image(current_depth_img)
#         next_depth_img = self._decode_depth_image(next_depth_img)
        
#         events = [np.load(os.path.join(sequence_dir, "events", f"{ts}.npy"), allow_pickle=True) for ts in event_ts]
#         events = np.concatenate(events, axis=0)
#         events = events[np.argsort(events["t"])]
#         events = events[events["t"] >= current_t]
#         events = events[events["t"] <= next_t]

#         event_frame = self._create_event_frame(current_depth_img.shape, events)
#         # make the event frame of shape 3, H, W from H, W, 3
#         event_frame = np.transpose(event_frame, (2, 0, 1))
#         event_frame = event_frame.astype(np.float32)
#         # add dimensions to 1, H, W
#         current_depth_img = np.expand_dims(current_depth_img, axis=0)
#         next_depth_img = np.expand_dims(next_depth_img, axis=0)
        
#         return {
#             "current_depth_img": current_depth_img,
#             "next_depth_img": next_depth_img,
#             "current_mask": current_mask,
#             "next_mask": next_mask,
#             "event_frame": event_frame
#         }

class CarlaSimDataset(Dataset):
    def __init__(self, data_dir):
        self.files = os.listdir(data_dir)
        self.data_dir = data_dir
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.files[idx]), allow_pickle=True)

        depth_frame = data["depth_img"]
        event_frame = data["event_frame"]
        next_depth_frame = data["next_depth_img"]

        event_frame = np.transpose(event_frame, (2, 0, 1)).astype(np.float32)
        depth_frame = np.expand_dims(depth_frame, axis=0)
        next_depth_frame = np.expand_dims(next_depth_frame, axis=0)
        
        return depth_frame, event_frame, next_depth_frame


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, split="train"):
        self.hdf5_file = hdf5_file
        self.split = split
        self.data = h5py.File(self.hdf5_file, "r")
        
    def __len__(self):
        if self.split == "train":
            return self.data["train"].shape[0]
        elif self.split == "test":
            return self.data["test"].shape[0]
        else:
            raise ValueError("Invalid split. Choose 'train' or 'test'.")
    
    def __getitem__(self, idx):
        depth_frame, next_depth_frame, event_frame = self.data[self.split][idx]
        depth_frame = np.expand_dims(depth_frame, axis=0)
        next_depth_frame = np.expand_dims(next_depth_frame, axis=0)
        event_frame = np.expand_dims(event_frame, axis=0)
        
        return depth_frame, event_frame, next_depth_frame
    
    def close(self):
        self.data.close()
        

class HDF5EVDataset(Dataset):
    def __init__(self, hdf5_file, split="train"):
        self.hdf5_file = hdf5_file
        self.split = split
        self.data = h5py.File(self.hdf5_file, "r")
        
    def __len__(self):
        if self.split == "train":
            return self.data["train"].shape[0]
        elif self.split == "test":
            return self.data["test"].shape[0]
        else:
            raise ValueError("Invalid split. Choose 'train' or 'test'.")
    
    def __getitem__(self, idx):
        depth_frame = self.data[self.split][idx][0]
        next_depth_frame = self.data[self.split][idx][1]
        event_frame = self.data[self.split][idx][2:]
        depth_frame = np.expand_dims(depth_frame, axis=0)
        next_depth_frame = np.expand_dims(next_depth_frame, axis=0)
        
        return depth_frame, event_frame, next_depth_frame
    
    def close(self):
        self.data.close()
        

class HDF5DatasetV2(Dataset):
    def __init__(self, hdf5_file, split="train", mask=False):
        self.hdf5_file = hdf5_file
        self.split = split
        self.mask = mask
        self.data = h5py.File(self.hdf5_file, "r")
        
    def __len__(self):
        if self.split in ["train", "test"]:
            return self.data.get(self.split).get("event_frames").shape[0]
        else:
            raise ValueError("Invalid split. Choose 'train' or 'test'.")
    
    def __getitem__(self, idx):
        event_frame = self.data[self.split]["event_frames"][idx]
        gt_depth_frame = self.data[self.split]["gt_depth_frames"][idx]
        prior_depth_frame = self.data[self.split]["prior_depth_frames"][idx]
        
        if self.mask:
            # gt_depth_frame = np.where(gt_depth_frame < 200, gt_depth_frame, 0)
            # prior_depth_frame = np.where(prior_depth_frame < 200, prior_depth_frame, 0)
            gt_depth_frame = np.clip(gt_depth_frame, 0, 200)
            prior_depth_frame = np.clip(prior_depth_frame, 0, 200)
        
        return prior_depth_frame, event_frame, gt_depth_frame
    
    def close(self):
        self.data.close()
        
        
class VTWDataset(Dataset):
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
        
        
class VTWVGDataset(Dataset):
    def __init__(self, hdf5_file: str, df: pd.DataFrame, n_bins: int):
        super().__init__()
        self.data = h5py.File(hdf5_file, "r")
        self.df = df
        self.n_bins = n_bins
        
    def to_voxel_grid(self, event_frames: np.ndarray):
        if len(event_frames) % self.n_bins != 0:
            empty_frames = np.zeros((self.n_bins - len(event_frames) % self.n_bins, 480, 640), dtype=event_frames.dtype)
            event_frames = np.concatenate((event_frames, empty_frames), axis=0)

        frames_per_bin = len(event_frames) // self.n_bins
        event_frames = np.reshape(event_frames, (self.n_bins, frames_per_bin, 480, 640))

        return event_frames.sum(axis=1)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        _, sequence, prior_depth_frame_id, gt_depth_frame_id, t1, t2 = self.df.iloc[idx]
        
        event_frames = self.data[f"{sequence}/events"][t1+1:t2+1]
        prior_depth_frame = self.data[f"{sequence}/depth_frames"][prior_depth_frame_id]
        gt_depth_frame = self.data[f"{sequence}/depth_frames"][gt_depth_frame_id]
        
        voxel_grid = self.to_voxel_grid(event_frames)

        return prior_depth_frame, voxel_grid, gt_depth_frame

    def close(self):
        self.data.close()