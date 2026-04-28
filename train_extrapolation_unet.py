import os
import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.configs import TrainingConfig
import utils.initializers as initializers
import utils.metrics as metrics
import utils.loss as loss_functions

from models.sunet import SUNet2EnResConv
from datetime import datetime


def print_summary(epoch: int, train_metrics: dict, test_metrics: dict):
    print(f"Epoch {epoch + 1}:")
    df = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"]).T
    df = df.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    print(df)
    
    
def train_model(model: nn.Module, train_dl: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, config: TrainingConfig):
    model.train()
    training_loss = []
    training_rmse = []
    
    with tqdm.tqdm(train_dl, desc="Training") as tepochs:
        for depth_frames, event_frames, gt_depth_frames in tepochs:
            depth_frames = depth_frames.to(dtype=torch.float32, device=device)
            event_frames = event_frames.to(dtype=torch.float32, device=device)
            gt_depth_frames = gt_depth_frames.to(dtype=torch.float32, device=device)
            
            # masking out the invalid depth values
            depth_frames[depth_frames > config.max_distance] = 0
            gt_depth_frames[gt_depth_frames > config.max_distance] = 0

            optimizer.zero_grad(set_to_none=True)

            pred_depth, _ = model(depth_frames, event_frames)

            loss = config.loss_fn(pred_depth, gt_depth_frames)
            rmse = config.eval_fn(pred_depth, gt_depth_frames)

            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())
            training_rmse.append(rmse)

            tepochs.set_postfix({
                "loss": loss.item(),
                "rmse": rmse
            })
            
    return {
        "dist_loss": np.mean(training_loss),
        "dist_rmse": np.mean(training_rmse)
    }


def test_model(model: nn.Module, test_dl: DataLoader, device: torch.device, config: TrainingConfig):
    model.eval()
    test_loss = []
    test_rmse = []
    
    with torch.no_grad():
        with tqdm.tqdm(test_dl, desc="Testing") as tepochs:
            for depth_frames, event_frames, gt_depth_frames in tepochs:
                depth_frames = depth_frames.to(dtype=torch.float32, device=device)
                event_frames = event_frames.to(dtype=torch.float32, device=device)
                gt_depth_frames = gt_depth_frames.to(dtype=torch.float32, device=device)
                
                # masking out the invalid depth values
                depth_frames[depth_frames > config.max_distance] = 0
                gt_depth_frames[gt_depth_frames > config.max_distance] = 0
                
                pred_depth, _ = model(depth_frames, event_frames)

                loss = config.loss_fn(pred_depth, gt_depth_frames)
                rmse = config.eval_fn(pred_depth, gt_depth_frames)

                test_loss.append(loss.item())
                test_rmse.append(rmse)

                tepochs.set_postfix({
                    "loss": loss.item(),
                    "rmse": rmse
                })
                
    return {
        "dist_loss": np.mean(test_loss),
        "dist_rmse": np.mean(test_rmse)
    }


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Available device is {device}")
    
    max_distance = 200.0  # in meters
    
    config= TrainingConfig(
        batch_size=8,
        max_cpu_count=16,
        epochs=50,
        learning_rate=0.001,
        weight_decay=0.0001,
        loss_fn=loss_functions.MultiLoss(device, weights={
            'dist_loss': 1.0,
            'ssim': 10.0,
            'gradient': 0.01,
            'normal': 1.0
        }),
        eval_fn=metrics.rmse,
        optimizer="nadam",
        mask=False,
        clip=False,
        max_distance=max_distance
    )
    
    
    metadata_file = "data/extrapolation/variable_interval_voxel_grids_v1_metadata.csv"
    hdf5_file = "data/extrapolation/variable_interval_voxel_grids_v1.h5"
    
    train_dataset, test_dataset = initializers.init_datasets(hdf5_file, metadata_file, config.clip)
    train_dl, test_dl = initializers.init_dataloaders(train_dataset, test_dataset, config)

    model=SUNet2EnResConv(n_channels=16, activation=nn.ReLU()).to(device)

    optimizer = initializers.init_optimizer(model, config)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6) # 1e-9 for 50 epochs
    config.lr_scheduler = schedular

    log_dir = f"runs/depth_extrapolation/{model._get_name()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    writer = SummaryWriter(log_dir=log_dir)
    config.save(os.path.join(log_dir, "config.pkl"))

    best_test_rmse = float('inf')
    best_test_loss = float('inf')

    for epoch in range(config.epochs):
        train_results = train_model(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            device=device,
            config=config
        )
        
        test_results = test_model(
            model=model,
            test_dl=test_dl,
            device=device,
            config=config
        )
        
        schedular.step()
        
        print_summary(epoch, train_results, test_results)
        
        for key, value in train_results.items():
            writer.add_scalar(f"{key}/train", value, epoch)
            
        for key, value in test_results.items():
            writer.add_scalar(f"{key}/test", value, epoch)
            
        if test_results["dist_rmse"] < best_test_rmse:
            best_test_rmse = test_results["dist_rmse"]
            torch.save(model.state_dict(), f"{log_dir}/best_model.pth")
            
        if test_results["dist_loss"] < best_test_loss:
            best_test_loss = test_results["dist_loss"]
            torch.save(model.state_dict(), f"{log_dir}/best_loss_model.pth")
            
        torch.save(model.state_dict(), f"{log_dir}/last_model.pth")

    writer.close()
    train_dataset.close()
    test_dataset.close()

if __name__ == "__main__":
    main()