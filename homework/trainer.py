import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn.functional as F


from .models import load_model, save_model
from .datasets.road_dataset import load_data

def compute_errors(pred_wp, gt_wp, mask):
    # mask: (B, n_waypoints)
    mask = mask.unsqueeze(-1)  # (B, n_waypoints, 1)
    diff = (pred_wp - gt_wp) * mask  # zero out ignored points

    # Longitudinal (x-axis) and lateral (y-axis) mean abs error
    lon_err = diff[..., 0].abs().sum() / mask.sum()
    lat_err = diff[..., 1].abs().sum() / mask.sum()
    return lon_err.item(), lat_err.item()

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # training loop
    for epoch in range(num_epoch):
        train_lon_errs, train_lat_errs = [], []

        model.train()
        for batch in train_data:
            track_left = batch["track_left"].to(device)      
            track_right = batch["track_right"].to(device)    
            waypoints = batch["waypoints"].to(device)        
            mask = batch["waypoints_mask"].to(device)

            # TODO: implement training step
            
            # Zero the gradients from the previous step
            optimizer.zero_grad()

            # Forward pass: compute predicted logits
            # Different models expect different inputs
            if model_name == "cnn_planner":
                image = batch["image"].to(device)  # (B, 3, H, W)
                pred_wp = model(image=image)
            else:  # mlp_planner or transformer_planner
                pred_wp = model(track_left=track_left, track_right=track_right)
                
            # Compute the loss
            loss_all = F.l1_loss(pred_wp, waypoints, reduction='none').mean(dim=-1)  # (B, n_waypoints)
            loss = (loss_all * mask).sum() / mask.sum()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Optimizer step: update model parameters
            optimizer.step()

            lon_err, lat_err = compute_errors(pred_wp, waypoints, mask)
            train_lon_errs.append(lon_err)
            train_lat_errs.append(lat_err)

            # Log training loss per iteration to TensorBoard
            logger.add_scalar("train_loss", loss.item(), global_step)
            
            global_step += 1

        # disable gradient computation and switch to evaluation mode
        val_lon_errs, val_lat_errs = [], []
        with torch.inference_mode():
            model.eval()
            for batch in val_data:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                # Forward pass
                pred_wp = model(track_left, track_right)

                lon_err, lat_err = compute_errors(pred_wp, waypoints, mask)
                val_lon_errs.append(lon_err)
                val_lat_errs.append(lat_err)
        
        # Average metrics
        epoch_train_lon = np.mean(train_lon_errs)
        epoch_train_lat = np.mean(train_lat_errs)
        epoch_val_lon = np.mean(val_lon_errs)
        epoch_val_lat = np.mean(val_lat_errs)

        logger.add_scalar("train_lon_err", epoch_train_lon, global_step)
        logger.add_scalar("train_lat_err", epoch_train_lat, global_step)
        logger.add_scalar("val_lon_err", epoch_val_lon, global_step)
        logger.add_scalar("val_lat_err", epoch_val_lat, global_step)

        # Log average train and val accuracy to TensorBoard at the end of the epoch
        # Use global_step to align epoch-level metrics with the last training iteration of the epoch


        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:02d}/{num_epoch:02d} "
                f"Train Lon: {epoch_train_lon:.3f} Lat: {epoch_train_lat:.3f} | "
                f"Val Lon: {epoch_val_lon:.3f} Lat: {epoch_val_lat:.3f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
