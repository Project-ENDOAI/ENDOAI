"""
Training script for intraoperative AI models.

This module provides functionality for training deep learning models that assist
in real-time during surgery, including sensor fusion, segmentation, instrument
tracking, and tissue classification.
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntraoperativeTrainer:
    """
    Base trainer class for intraoperative AI models.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialize the trainer with model and configuration.
        
        Args:
            model: PyTorch model to train
            config: Training configuration parameters
        """
        self.config = config
        self.model = model
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        optimizer_name = config.get("optimizer", "adam").lower()
        lr = config.get("learning_rate", 1e-4)
        weight_decay = config.get("weight_decay", 1e-5)
        
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            momentum = config.get("momentum", 0.9)
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup learning rate scheduler
        scheduler_name = config.get("lr_scheduler", None)
        if scheduler_name == "step":
            step_size = config.get("lr_step_size", 30)
            gamma = config.get("lr_gamma", 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "cosine":
            T_max = config.get("lr_T_max", 100)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        else:
            self.scheduler = None
        
        # Setup loss function
        self.criterion = self._get_loss_function()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}
        self.best_val_loss = float('inf')
        
        # Output directory
        self.output_dir = config.get("output_dir", os.path.join("output", "models", 
                                                              f"intraop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_loss_function(self) -> nn.Module:
        """
        Get the appropriate loss function based on configuration.
        
        Returns:
            PyTorch loss function
        """
        loss_name = self.config.get("loss", "cross_entropy").lower()
        
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        elif loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "l1":
            return nn.L1Loss()
        elif loss_name == "dice":
            # Custom Dice loss implementation for segmentation
            return DiceLoss()
        elif loss_name == "focal":
            # Custom Focal loss implementation for imbalanced classification
            gamma = self.config.get("focal_gamma", 2.0)
            alpha = self.config.get("focal_alpha", 0.25)
            return FocalLoss(gamma=gamma, alpha=alpha)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            
        Returns:
            Dictionary with training results and metrics
        """
        num_epochs = self.config.get("epochs", 100)
        log_interval = self.config.get("log_interval", 10)
        save_best = self.config.get("save_best", True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss, train_metrics = self._train_epoch(train_loader, epoch, log_interval)
            self.train_losses.append(train_loss)
            
            # Store training metrics
            for metric_name, value in train_metrics.items():
                if metric_name not in self.train_metrics:
                    self.train_metrics[metric_name] = []
                self.train_metrics[metric_name].append(value)
            
            # Validation if loader provided
            if val_loader is not None:
                val_loss, val_metrics = self._validate(val_loader, epoch)
                self.val_losses.append(val_loss)
                
                # Store validation metrics
                for metric_name, value in val_metrics.items():
                    if metric_name not in self.val_metrics:
                        self.val_metrics[metric_name] = []
                    self.val_metrics[metric_name].append(value)
                
                # Save best model
                if save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(f"best_model.pth", epoch, val_loss)
                    logger.info(f"Saved best model with validation loss: {val_loss:.6f}")
            
            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Save final model
        self._save_checkpoint("final_model.pth", num_epochs, self.train_losses[-1])
        
        # Training summary
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Plot training curves
        self._plot_training_curves()
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "best_val_loss": self.best_val_loss,
            "training_time": total_time,
            "output_dir": self.output_dir
        }
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int, log_interval: int) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            log_interval: How often to log progress
            
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.train()
        total_loss = 0.0
        metrics = {}
        
        for batch_idx, batch in enumerate(train_loader):
            # Process batch - implementation depends on data format
            inputs, targets = self._process_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % log_interval == 0:
                logger.info(f"Epoch {epoch}/{self.config.get('epochs', 100)} "
                          f"[{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(train_loader, "train")
        
        logger.info(f"Epoch {epoch} - Average training loss: {avg_loss:.6f}")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.6f}")
        
        return avg_loss, metrics
    
    def _validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Process batch
                inputs, targets = self._process_batch(batch)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Track loss
                total_loss += loss.item()
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(val_loader, "val")
        
        logger.info(f"Epoch {epoch} - Validation loss: {avg_loss:.6f}")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.6f}")
        
        return avg_loss, metrics
    
    def _process_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of data.
        
        Args:
            batch: Batch of data from DataLoader
            
        Returns:
            Tuple of (inputs, targets) tensors
        """
        # Default implementation assumes batch is a tuple of (inputs, targets)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            raise ValueError("Batch format not supported by default _process_batch implementation")
        
        # Move to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        return inputs, targets
    
    def _calculate_metrics(self, data_loader: DataLoader, phase: str) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            data_loader: DataLoader for data
            phase: 'train' or 'val'
            
        Returns:
            Dictionary of metric names and values
        """
        # Placeholder - should be overridden by subclasses for specific metrics
        return {}
    
    def _save_checkpoint(self, filename: str, epoch: int, loss: float) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            epoch: Current epoch
            loss: Current loss value
        """
        checkpoint_path = os.path.join(self.output_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with loaded checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def _plot_training_curves(self) -> None:
        """
        Plot training and validation curves.
        """
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics plot (if any)
        if self.train_metrics or self.val_metrics:
            plt.subplot(1, 2, 2)
            for name, values in self.train_metrics.items():
                plt.plot(values, label=f'Train {name}')
            for name, values in self.val_metrics.items():
                plt.plot(values, label=f'Val {name}')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.title('Training Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))


class SegmentationTrainer(IntraoperativeTrainer):
    """
    Trainer for intraoperative segmentation models.
    """
    
    def _calculate_metrics(self, data_loader: DataLoader, phase: str) -> Dict[str, float]:
        """Calculate segmentation-specific metrics like Dice, IoU, etc."""
        self.model.eval()
        dice_sum = 0.0
        iou_sum = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = self._process_batch(batch)
                outputs = self.model(inputs)
                
                # Convert outputs to binary predictions
                preds = torch.sigmoid(outputs) > 0.5
                
                # Calculate metrics
                dice_sum += self._dice_coefficient(preds, targets).item()
                iou_sum += self._iou(preds, targets).item()
                count += inputs.size(0)
        
        return {
            'dice': dice_sum / count,
            'iou': iou_sum / count
        }
    
    def _dice_coefficient(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Dice coefficient."""
        smooth = 1.0
        
        if preds.dim() > 2:
            # For 3D or 4D tensors, flatten all dimensions except batch
            preds = preds.contiguous().view(preds.size(0), -1)
            targets = targets.contiguous().view(targets.size(0), -1)
            
        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean()
    
    def _iou(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate IoU (Intersection over Union)."""
        smooth = 1.0
        
        if preds.dim() > 2:
            preds = preds.contiguous().view(preds.size(0), -1)
            targets = targets.contiguous().view(targets.size(0), -1)
        
        intersection = (preds * targets).sum(dim=1)
        total = (preds + targets).sum(dim=1)
        union = total - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()


class SensorFusionTrainer(IntraoperativeTrainer):
    """
    Trainer for sensor fusion models.
    """
    
    def _process_batch(self, batch: Any) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Process a batch with multiple sensor inputs.
        
        Args:
            batch: Batch of data from DataLoader with multiple sensor inputs
            
        Returns:
            Tuple of (sensor_inputs_dict, targets)
        """
        # This implementation assumes batch is a dictionary or tuple with sensor data
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            # If batch contains a tuple of (sensor_dict, targets)
            sensor_inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            # If batch is a dictionary with 'inputs' and 'targets' keys
            sensor_inputs = {k: v for k, v in batch.items() if k != 'targets'}
            targets = batch['targets']
        else:
            raise ValueError("Batch format not supported by SensorFusionTrainer")
        
        # Move tensors to device
        if isinstance(sensor_inputs, dict):
            for key in sensor_inputs:
                if isinstance(sensor_inputs[key], torch.Tensor):
                    sensor_inputs[key] = sensor_inputs[key].to(self.device)
        else:
            sensor_inputs = sensor_inputs.to(self.device)
            
        targets = targets.to(self.device)
        
        return sensor_inputs, targets


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to raw inputs if not already done
        if not ((inputs >= 0) & (inputs <= 1)).all():
            inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Dice score
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss (1 - Dice score)
        return 1.0 - dice_score


class FocalLoss(nn.Module):
    """Focal loss for handling imbalanced classes."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Get standard BCE loss
        bce_loss = self.bce(inputs, targets)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        
        # Apply alpha weighting
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Calculate focal loss
        focal_loss = alpha_weight * (1 - p_t) ** self.gamma * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def main():
    """
    Main function for running the training script from command line.
    """
    import argparse
    from importlib import import_module
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train intraoperative models for surgical assistance")
    parser.add_argument('--config', type=str, required=True, help='Path to training configuration file')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing training data')
    parser.add_argument('--model-class', type=str, required=True, 
                        help='Model class to train, e.g., "endoai.src.intraoperative.models.SegmentationModel"')
    parser.add_argument('--trainer', type=str, default='segmentation', 
                        choices=['segmentation', 'sensor_fusion', 'base'],
                        help='Trainer class to use')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Directory to save model and outputs (default: auto-generated)')
    args = parser.parse_args()
    
    # Load configuration
    try:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override config with command line arguments
        if args.output_dir:
            config['output_dir'] = args.output_dir
        config['data_dir'] = args.data_dir
        
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        return
    
    # Import model class dynamically
    try:
        module_path, class_name = args.model_class.rsplit('.', 1)
        module = import_module(module_path)
        ModelClass = getattr(module, class_name)
        model = ModelClass(**config.get('model_params', {}))
        logger.info(f"Loaded model: {args.model_class}")
    except Exception as e:
        logger.error(f"Error loading model class: {e}")
        return
    
    # Create data loaders
    try:
        from endoai.src.intraoperative.data_loader import create_dataloaders
        train_loader, val_loader = create_dataloaders(config)
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        return
    
    # Select trainer class
    if args.trainer == 'segmentation':
        trainer = SegmentationTrainer(model, config)
    elif args.trainer == 'sensor_fusion':
        trainer = SensorFusionTrainer(model, config)
    else:
        trainer = IntraoperativeTrainer(model, config)
    
    # Train model
    try:
        results = trainer.train(train_loader, val_loader)
        logger.info(f"Training completed. Model saved to {results['output_dir']}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
