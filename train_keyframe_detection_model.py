import os
import tqdm
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
import utils.initializers as initializers

from utils.configs import KFDTrainingConfig
from models.cnn import L3CNNV4
from datetime import datetime


def print_summary(epoch: int, train_metrics: dict, test_metrics: dict):
    print(f"Epoch {epoch + 1}:")
    df = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"]).T
    df = df.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    print(df)


def train_model(model: nn.Module, train_dl: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, config: KFDTrainingConfig):
    model.train()
    training_loss = []
    training_f1 = []
    
    eval_fn = config.eval_fn.to(device)

    with tqdm.tqdm(train_dl, desc="Training") as tepochs:
        for event_frames, scores in tepochs:
            event_frames = event_frames.to(dtype=torch.float32, device=device)
            scores = scores.to(dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)

            pred_logits = model(event_frames)
            pred_scores = torch.sigmoid(pred_logits)

            loss = config.loss_fn(pred_logits, scores)
            f1 = eval_fn(pred_scores, scores)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            training_loss.append(loss.item())
            training_f1.append(f1.item())

            tepochs.set_postfix(loss=loss.item(), f1=f1.item())

    return {
        "loss": np.mean(training_loss),
        "f1": np.mean(training_f1)
    }


def test_model(model: nn.Module, test_dl: DataLoader, device: torch.device, config: KFDTrainingConfig):
    model.eval()
    test_loss = []
    test_f1 = []
    
    eval_fn = config.eval_fn.to(device)

    with torch.no_grad():
        with tqdm.tqdm(test_dl, desc="Testing") as tepochs:
            for event_frames, scores in tepochs:
                event_frames = event_frames.to(dtype=torch.float32, device=device)
                scores = scores.to(dtype=torch.float32, device=device)

                pred_logits = model(event_frames)
                pred_scores = torch.sigmoid(pred_logits)

                loss = config.loss_fn(pred_logits, scores)
                f1 = eval_fn(pred_scores, scores)

                test_loss.append(loss.item())
                test_f1.append(f1.item())

                tepochs.set_postfix(loss=loss.item(), f1=f1.item())

    return {
        "loss": np.mean(test_loss),
        "f1": np.mean(test_f1)
    }


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Available device is {device}")

    config = KFDTrainingConfig(
        batch_size=64,
        max_cpu_count=16,
        epochs=20,
        learning_rate=1e-4,
        weight_decay=0.01,
        loss_fn=nn.BCEWithLogitsLoss(),
        eval_fn=BinaryF1Score(threshold=0.5).to(device),
        optimizer="nadam",
        downsample_factor=1,
        input_size=(640, 480)
    )

    hdf5_file = "data/slicing/slicing_dataset_v1.h5"

    train_dataset, test_dataset = initializers.init_keyframe_datasets(hdf5_file)
    train_dl, test_dl = initializers.init_keyframe_dataloaders(train_dataset, test_dataset, config)

    model = L3CNNV4(
        in_channels=1,
        down_sample_factor=config.downsample_factor,
        input_size=config.input_size
    ).to(device)

    optimizer = initializers.init_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    log_dir = f"checkpoints/{model._get_name()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    config.save(os.path.join(log_dir, "config.pkl"))

    best_val_f1 = 0.0
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        train_results = train_model(model, train_dl, optimizer, device, config)
        test_results = test_model(model, test_dl, device, config)

        scheduler.step(test_results["loss"])

        print_summary(epoch, train_results, test_results)

        for key, value in train_results.items():
            writer.add_scalar(f"{key}/train", value, epoch + 1)

        for key, value in test_results.items():
            writer.add_scalar(f"{key}/test", value, epoch + 1)

        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch + 1)

        if test_results["f1"] > best_val_f1:
            best_val_f1 = test_results["f1"]
            torch.save(model.state_dict(), f"{log_dir}/best_f1_score.pth")

        if test_results["loss"] < best_val_loss:
            best_val_loss = test_results["loss"]
            torch.save(model.state_dict(), f"{log_dir}/best_loss.pth")

        torch.save(model.state_dict(), f"{log_dir}/last.pth")

    writer.close()
    train_dataset.close()
    test_dataset.close()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
