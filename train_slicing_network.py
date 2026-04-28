import os
import tqdm
import torch
import pickle
import h5py
import numpy as np
import torchvision.models as models    
import torch.nn.functional as F

from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryF1Score
from datetime import datetime


class HDF5SlicingDataset(Dataset):
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


 
class L3CNNV4(nn.Module):
    def __init__(self, in_channels=1, down_sample_factor=2, input_size=(640, 480)):
        super(L3CNNV4, self).__init__()
        self.fc_features = int(4 * (input_size[0] / down_sample_factor / 8) * (input_size[1] / down_sample_factor / 8))

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4)

        self.fc1 = nn.Linear(self.fc_features, 128)
        self.fc2 = nn.Linear(128, 1)

        self.downsample = nn.AvgPool2d(kernel_size=down_sample_factor, stride=down_sample_factor)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.activation = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.downsample(x)
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Trainer:
    def __init__(self, h5py_file, learning_rate=0.0001, epochs=20, batch_size=64, num_workers=48, weight_decay=0.02, downsample_factor=1):
        self.h5py_file = h5py_file
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.downsample_factor = downsample_factor
        self.input_size = (640, 480)
        self.log_dir = None
        self.loss_fn = None
        self.description = ""
        
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
        
    def set_description(self, description):
        self.description = description
        
    def get_log_dir(self, base_dir, model_name):
        if self.log_dir is None:
            self.log_dir = os.path.join(base_dir, f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(self.log_dir, exist_ok=True)
        return self.log_dir
        
    def get_datasets(self):
        train_dataset = HDF5SlicingDataset(self.h5py_file, split="train")
        test_dataset = HDF5SlicingDataset(self.h5py_file, split="test")
        return train_dataset, test_dataset
    
    def get_dataloaders(self):
        train_dataset, test_dataset = self.get_datasets()
        train_dl = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, prefetch_factor=2)
        test_dl = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False, prefetch_factor=2)
        return train_dl, test_dl
    
    def get_optimizer(self, model):
        optimizer = optim.NAdam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
        return optimizer, scheduler
      
    def save(self, filepath):
        pickle.dump(self, open(filepath, "wb"))
        
    @staticmethod
    def load(filepath):
        return pickle.load(open(filepath, "rb"))


def training(device: torch.device, config: Trainer):
    model = L3CNNV4(
        in_channels=1,
        down_sample_factor=config.downsample_factor,
        input_size=config.input_size
    ).to(device)
    
    train_dl, test_dl = config.get_dataloaders()
    optimizer, scheduler = config.get_optimizer(model)

    eval_fn = BinaryF1Score(threshold=0.5).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    
    log_dir = config.get_log_dir(base_dir="checkpoints/", model_name=model._get_name())
    writer = SummaryWriter(log_dir=log_dir)

    config.set_loss_fn(loss_fn._get_name())
    config.save(os.path.join(log_dir, "config.pkl"))
    
    best_val_f1_score = 0
    best_val_loss = np.inf

    avg_training_loss = 0
    avg_training_f1_score = 0
    avg_validation_loss = 0
    avg_validation_f1_score = 0

    for epoch in range(config.epochs):
        # Training phase
        model.train()

        training_loss = []
        training_f1_score = []

        print(f"Epoch {epoch+1}/{config.epochs}")
        
        with tqdm.tqdm(train_dl, desc="Training") as tepochs:
            for event_frames, scores in tepochs:
                event_frames = event_frames.to(dtype=torch.float32, device=device)
                scores = scores.to(dtype=torch.float32, device=device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
            
                # forward + backward + optimize
                pred_logits = model(event_frames)
                pred_scores = torch.sigmoid(pred_logits)
                
                loss = loss_fn(pred_logits, scores)
                f1_score = eval_fn(pred_scores, scores)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                
                tepochs.set_postfix(loss=loss.item(), f1=f1_score.item())
                
                training_loss.append(loss.item())
                training_f1_score.append(f1_score.item())

        avg_training_loss = np.mean(training_loss)
        avg_training_f1_score = np.mean(training_f1_score)

        # Validation phase
        model.eval()

        validation_loss = []
        validation_f1_score = []

        with torch.no_grad():   
            with tqdm.tqdm(test_dl, desc="Validation") as tepochs:
                for event_frames, scores in tepochs:
                    event_frames = event_frames.to(dtype=torch.float32, device=device)
                    scores = scores.to(dtype=torch.float32, device=device)
                    
                    pred_logits = model(event_frames)
                    pred_scores = torch.sigmoid(pred_logits)
                    
                    loss = loss_fn(pred_logits, scores)
                    f1_score = eval_fn(pred_scores, scores)

                    tepochs.set_postfix(loss=loss.item(), f1=f1_score.item())

                    validation_loss.append(loss.item())
                    validation_f1_score.append(f1_score.item())
                
        avg_validation_loss = np.mean(validation_loss)
        avg_validation_f1_score = np.mean(validation_f1_score)

        print(f"Train Loss: {avg_training_loss:.4f}, Train F1-Score: {avg_training_f1_score:.4f}")
        print(f"Valid Loss: {avg_validation_loss:.4f}, Valid F1-Score: {avg_validation_f1_score:.4f}")
        
        writer.add_scalar("loss/train", avg_training_loss, epoch + 1)
        writer.add_scalar("loss/valid", avg_validation_loss, epoch + 1)
        writer.add_scalar("f1/train", avg_training_f1_score, epoch + 1)
        writer.add_scalar("f1/valid", avg_validation_f1_score, epoch + 1)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch + 1)

        if avg_validation_f1_score > best_val_f1_score:
            best_val_f1_score = avg_validation_f1_score
            torch.save(model.state_dict(), f"{writer.log_dir}/best_f1_score.pth")
            
        if avg_validation_loss < best_val_loss:
            best_val_loss = avg_validation_loss
            torch.save(model.state_dict(), f"{writer.log_dir}/best_loss.pth")
                    
        torch.save(model.state_dict(), f"{writer.log_dir}/last.pth")

        scheduler.step(avg_validation_loss) # only if using ReduceLROnPlateau

    writer.close()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Available device is {device}")
    
    config = Trainer(
        h5py_file="data/slicing/slicing_dataset_v1.h5",
        learning_rate=1e-4,
        epochs=20,
        batch_size=64,
        num_workers=16,
        weight_decay=0.01
    )

    training(device, config)
    
    
if __name__ == "__main__":
    main()