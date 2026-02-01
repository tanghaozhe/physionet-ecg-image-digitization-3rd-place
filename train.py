import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd
import numpy as np

from configs.stage2 import Stage2Config_4352x1696, Stage2HRNetConfig_4352x1696, Stage2ConvNeXtV2Config_4352x1696, Stage2EfficientNetV2Config_4352x1696, Stage2ResNeStConfig_4352x1696
from models.stage2_net import Stage2Net, Stage2HRNet, Stage2ConvNeXtV2, Stage2EfficientNetV2, Stage2ResNeSt
from datasets.stage2_dataset import Stage2Dataset
from utils.common import set_seed, load_checkpoint
from utils.metrics import SegmentationMetrics
from utils.postprocess import mask_to_prediction, csv_to_kaggle_format
from utils.kaggle_metric import score
import wandb


DEBUG = False
DEBUG_SAMPLES = 100

def train_one_epoch(model, loader, optimizer, scheduler, cfg, epoch, scaler=None):
    model.train()
    total_loss = 0
    metrics = SegmentationMetrics(threshold=0.5)

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):

        if cfg.use_amp and scaler is not None:
            with autocast('cuda'):
                output = model(batch)
                loss = output['pixel_loss']
                loss = loss / cfg.accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

        else:
            output = model(batch)
            loss = output['pixel_loss']
            loss = loss / cfg.accumulation_steps
            loss.backward()

            if (batch_idx + 1) % cfg.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step() 

        # total_loss += loss.item()
        total_loss += loss.item() * cfg.accumulation_steps

        with torch.no_grad():
            pred = output['pixel'].detach().cpu()
            target = batch['pixel'].cpu()
            metrics.update(pred, target)

        pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

    avg_loss = total_loss / len(loader)
    train_metrics = metrics.get_metrics()

    return avg_loss, train_metrics

def validate(model, loader, cfg, epoch):
    model.eval()
    total_loss = 0
    metrics = SegmentationMetrics(threshold=0.1)

    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            if cfg.use_amp:
                with autocast('cuda'):
                    output = model(batch)
                    loss = output['pixel_loss']
            else:
                output = model(batch)
                loss = output['pixel_loss']

            total_loss += loss.item()

            pred = output['pixel'].detach().cpu()
            target = batch['pixel'].cpu()
            metrics.update(pred, target)

    avg_loss = total_loss / len(loader)
    val_metrics = metrics.get_metrics()

    return avg_loss, val_metrics

def train_single_fold(cfg, fold):
    print(f'\n{"="*60}')
    print(f'Training Fold {fold}/{cfg.num_folds} - HIGH RESOLUTION (4400×1700)')
    print(f'{"="*60}')
    print(f'Rectified dir: {cfg.rectified_dir}')
    print(f'Mask dir: {cfg.mask_dir}')
    print(f'Image size: {cfg.crop_x_range[1]}×{cfg.crop_y_range[1]}')
    print(f'Batch size: {cfg.batch_size}')
    print(f'Accumulation steps: {cfg.accumulation_steps}')
    print(f'AMP (Mixed Precision): {"Enabled" if cfg.use_amp else "Disabled"}')
    if DEBUG:
        print(f'DEBUG MODE: Training on {DEBUG_SAMPLES} samples only')
    print()

    fold_checkpoint_dir = os.path.join(cfg.checkpoint_dir, f'fold_{fold}')
    os.makedirs(fold_checkpoint_dir, exist_ok=True)

    # model = Stage2Net(cfg).to(cfg.device)
    # model = Stage2HRNet(cfg).to(cfg.device)
    model = Stage2ConvNeXtV2(cfg).to(cfg.device)
    # model = Stage2EfficientNetV2(cfg).to(cfg.device)
    # model = Stage2ResNeSt(cfg).to(cfg.device)
    
    if cfg.checkpoint_path is not None:
        checkpoint_path = cfg.checkpoint_path.format(fold)
        model = load_checkpoint(model, checkpoint_path)

    train_dataset = Stage2Dataset(cfg, split='train', fold=fold)
    val_dataset = Stage2Dataset(cfg, split='val', fold=fold)

    if DEBUG:
        train_dataset = Subset(train_dataset, range(min(DEBUG_SAMPLES, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(DEBUG_SAMPLES, len(val_dataset))))

    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=2 if cfg.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=cfg.num_epochs, eta_min=1e-5,
    # )

    steps_per_epoch = (len(train_loader) + cfg.accumulation_steps - 1) // cfg.accumulation_steps

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,  
        epochs=cfg.num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.05,          
        anneal_strategy='cos',  
        div_factor=25.0, 
        final_div_factor=1000.0  
    )

    scaler = GradScaler('cuda') if cfg.use_amp else None

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f'stage2_highres_fold{fold}',
            config={
                'fold': fold,
                'stage': 'stage2_highres',
                'batch_size': cfg.batch_size,
                'learning_rate': cfg.learning_rate,
                'num_epochs': cfg.num_epochs,
                'use_amp': cfg.use_amp,
            }
        )

    best_loss = float('inf')

    for epoch in range(cfg.num_epochs):
        print(f'\nFold {fold}, Epoch {epoch}/{cfg.num_epochs}')

        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, epoch, scaler)
        val_loss, val_metrics = validate(model, val_loader, cfg, epoch)

        current_lr = optimizer.param_groups[0]['lr']

        print(f'Train Loss: {train_loss:.4f} | IoU: {train_metrics["iou"]:.4f} | Dice: {train_metrics["dice"]:.4f} | F1: {train_metrics["f1"]:.4f}')
        print(f'Val   Loss: {val_loss:.4f} | IoU: {val_metrics["iou"]:.4f} | Dice: {val_metrics["dice"]:.4f} | F1: {val_metrics["f1"]:.4f}')

        if cfg.use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_iou': train_metrics['iou'],
                'train_dice': train_metrics['dice'],
                'train_accuracy': train_metrics['accuracy'],
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'train_f1': train_metrics['f1'],
                'val_iou': val_metrics['iou'],
                'val_dice': val_metrics['dice'],
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'lr': current_lr,
            }
            wandb.log(log_dict)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(fold_checkpoint_dir, 'best.pth'))
            print(f'Saved best model with loss: {best_loss:.4f}')

        if epoch % cfg.save_frequency == 0:
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(fold_checkpoint_dir, f'epoch_{epoch:04d}.pth'))

    torch.save({
        'fold': fold,
        'epoch': cfg.num_epochs - 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(fold_checkpoint_dir, 'last.pth'))

    if cfg.use_wandb:
        wandb.finish()

    return best_loss

def main():
    cfg = Stage2ConvNeXtV2Config_4352x1696() # Stage2EfficientNetV2Config_4352x1696() #Stage2HRNetConfig_4352x1696() #Stage2ConvNeXtV2Config_4352x1696() # Stage2Config_4352x3392()
    set_seed(cfg.seed)

    print(f'\n{"="*60}')
    print('Stage 2 High-Resolution Training')
    print(f'{"="*60}')
    print(f'Resolution: 4400×1700 (2× wider than original)')
    print(f'Rectified dir: {cfg.rectified_dir}')
    print(f'Mask dir: {cfg.mask_dir}')
    print(f'Output dir: {cfg.stage_output_dir}')
    print(f'Training folds: {cfg.train_folds}')
    print(f'{"="*60}\n')

    fold_scores = []
    for fold in cfg.train_folds:
        best_loss = train_single_fold(cfg, fold)
        fold_scores.append(best_loss)
        print(f'\nFold {fold} completed with best loss: {best_loss:.4f}')

    mean_loss = sum(fold_scores) / len(fold_scores)
    std_loss = (sum((x - mean_loss)**2 for x in fold_scores) / len(fold_scores))**0.5
    min_loss = min(fold_scores)
    max_loss = max(fold_scores)

    print(f'\n{"="*60}')
    print('K-Fold Training Completed')
    print(f'{"="*60}')
    print(f'Fold scores: {fold_scores}')
    print(f'Mean: {mean_loss:.4f}')
    print(f'Std: {std_loss:.4f}')
    print(f'Min: {min_loss:.4f}')
    print(f'Max: {max_loss:.4f}')

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name='stage2_highres_cv_summary',
            config={'num_folds': cfg.num_folds}
        )

        for idx, (fold, loss) in enumerate(zip(cfg.train_folds, fold_scores)):
            wandb.log({f'fold{fold}/best_val_loss': loss})

        wandb.summary['mean_val_loss'] = mean_loss
        wandb.summary['std_val_loss'] = std_loss
        wandb.summary['min_val_loss'] = min_loss
        wandb.summary['max_val_loss'] = max_loss
        wandb.summary['num_folds'] = len(fold_scores)

        wandb.finish()


    print('Training completed!')

if __name__ == '__main__':
    main()
