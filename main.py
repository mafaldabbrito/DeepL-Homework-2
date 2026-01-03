"""
RNA Binding Protein (RBP) Prediction using Convolutional Neural Networks

This script implements a CNN model for predicting RNA-protein binding affinities.
The CNN acts as a motif detector, identifying local sequence patterns associated 
with high binding affinity, similar to how RBPs scan RNA for binding sites.

Architecture:
    1. Convolutional Layer: Detects local motifs (binding patterns)
    2. Pooling Layer: Aggregates motif information (configurable strategy)
    3. Fully Connected Layers: Maps motif features to binding intensity
    4. Dropout: Regularization to prevent overfitting

Key Features:
    - Multiple pooling strategies (global max, local max, none)
    - Hyperparameter tuning for optimal performance
    - Masked loss to handle missing data points
    - Spearman correlation for evaluation
"""

import argparse
import json
import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

from utils import (
    load_rnacompete_data,
    RNACompeteLoader,  # Import the loader class directly
    configure_seed, 
    masked_mse_loss, 
    masked_spearman_correlation,
    plot,
    dataset_summary,
)
from config import RNAConfig


# ============================================================================
# STEP 1: Define the CNN Model Architecture
# ============================================================================

class RNABindingCNN(nn.Module):
    """
    Convolutional Neural Network for RNA Binding Prediction.
    
    The model uses 1D convolutions to detect sequence motifs, followed by
    pooling to aggregate information, and fully connected layers for prediction.
    
    Args:
        num_filters: Number of convolutional filters (motif detectors)
        kernel_size: Size of each filter (motif length to detect)
        pooling_type: Type of pooling ('global_max', 'local_max', 'none')
        dropout_rate: Dropout probability for regularization
        input_channels: Number of input channels (4 for one-hot RNA)
        seq_length: Length of input sequence (41 for RNAcompete)
    """
    
    def __init__(
        self, 
        num_filters: int = 64,
        kernel_size: int = 8,
        pooling_type: str = 'global_max',
        dropout_rate: float = 0.3,
        input_channels: int = 4,
        seq_length: int = 41
    ):
        super(RNABindingCNN, self).__init__()
        
        self.pooling_type = pooling_type
        self.seq_length = seq_length
        
        # Convolutional Layer: Each filter acts as a motif detector
        # Input: (batch, 4, 41) -> Output: (batch, num_filters, L_out)
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2  # Keep sequence length similar
        )
        
        # Pooling Layer: Aggregates motif information
        self.pool_size = 2  # For local max pooling
        if pooling_type == 'local_max':
            self.pool = nn.MaxPool1d(kernel_size=self.pool_size)
            # Calculate the output length after convolution and pooling
            conv_out_len = seq_length  # With padding='same'
            pool_out_len = conv_out_len // self.pool_size
            fc_input_size = num_filters * pool_out_len
        elif pooling_type == 'global_max':
            # Global max pooling will reduce to (batch, num_filters)
            fc_input_size = num_filters
        else:  # 'none'
            # No pooling, keep all positions
            fc_input_size = num_filters * seq_length
        
        # Fully Connected Layers: Map motif features to binding intensity
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output: single binding intensity value
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, 4)
        
        Returns:
            predictions: Tensor of shape (batch_size, 1)
        """
        # Input shape: (batch, 41, 4)
        # Conv1d expects: (batch, channels, length)
        x = x.transpose(1, 2)  # -> (batch, 4, 41)
        
        # Apply Conv -> ReLU
        x = F.relu(self.conv1(x))  # -> (batch, num_filters, ~41)
        
        # Pooling: Aggregate motif information
        if self.pooling_type == 'global_max':
            # Global max: Take max across all positions
            # Captures if motif is present anywhere in sequence
            x = torch.max(x, dim=2)[0]  # -> (batch, num_filters)
        elif self.pooling_type == 'local_max':
            # Local max: Preserve some positional information
            x = self.pool(x)  # -> (batch, num_filters, ~20)
            x = torch.flatten(x, 1)  # Flatten
        else:  # 'none'
            # No pooling: Preserve full positional information
            x = torch.flatten(x, 1)  # Flatten
        
        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)  # -> (batch, 1)
        
        return x

# ============================================================================
# STEP 2: Training Function
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The CNN model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        device: Device to run on (CPU or CUDA)
    
    Returns:
        avg_loss: Average training loss
        avg_corr: Average Spearman correlation
    """
    model.train()
    total_loss = 0.0
    total_corr = 0.0
    num_batches = 0
    
    for batch_idx, (sequences, targets, masks) in enumerate(train_loader):
        # Move data to device
        sequences = sequences.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(sequences)
        
        # Compute masked loss (ignores NaN values)
        loss = masked_mse_loss(predictions, targets, masks)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            corr = masked_spearman_correlation(predictions, targets, masks)
        
        total_loss += loss.item()
        total_corr += corr.item()
        num_batches += 1
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                  f"Loss = {loss.item():.4f}, Corr = {corr.item():.4f}")
    
    avg_loss = total_loss / num_batches
    avg_corr = total_corr / num_batches
    
    return avg_loss, avg_corr


# ============================================================================
# STEP 3: Evaluation Function
# ============================================================================

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model on validation or test set.
    
    Args:
        model: The CNN model
        data_loader: DataLoader for evaluation data
        device: Device to run on
    
    Returns:
        avg_loss: Average loss
        avg_corr: Average Spearman correlation
    """
    model.eval()
    total_loss = 0.0
    total_corr = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for sequences, targets, masks in data_loader:
            # Move data to device
            sequences = sequences.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            # Forward pass
            predictions = model(sequences)
            
            # Compute metrics
            loss = masked_mse_loss(predictions, targets, masks)
            corr = masked_spearman_correlation(predictions, targets, masks)
            
            total_loss += loss.item()
            total_corr += corr.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_corr = total_corr / num_batches
    
    return avg_loss, avg_corr


# ============================================================================
# STEP 4: Full Training Loop
# ============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    save_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Train the model for multiple epochs with validation.
    
    Args:
        model: The CNN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run on
        save_path: Path to save the best model (optional)
    
    Returns:
        history: Dictionary containing training metrics
    """
    print(f"\n{'='*70}")
    print(f"Starting Training on {device}")
    print(f"{'='*70}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track metrics
    history = {
        'train_loss': [],
        'train_corr': [],
        'val_loss': [],
        'val_corr': []
    }
    
    best_val_corr = -1.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"{'-'*70}")
        
        # Train for one epoch
        train_loss, train_corr = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate on validation set
        val_loss, val_corr = evaluate(model, val_loader, device)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_corr'].append(train_corr)
        history['val_loss'].append(val_loss)
        history['val_corr'].append(val_corr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Corr: {train_corr:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Corr:   {val_corr:.4f}")
        
        # Save best model
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_corr': val_corr,
                }, save_path)
                print(f"  >>> New best model saved! (Val Corr: {val_corr:.4f})")
    
    print(f"\n{'='*70}")
    print(f"Training Complete! Best Val Correlation: {best_val_corr:.4f}")
    print(f"{'='*70}\n")
    
    return history


# ============================================================================
# STEP 5: Hyperparameter Tuning
# ============================================================================

def hyperparameter_search(
    protein_name: str,
    config: RNAConfig,
    device: torch.device,
    batch_size: int = 64,
    num_epochs: int = 20
) -> Tuple[Dict, Dict]:
    """
    Perform grid search over hyperparameters.
    
    This function tests different combinations of:
    - Filter sizes (32, 64, 128)
    - Kernel sizes (6, 8, 10)
    - Pooling types (global_max, local_max, none)
    - Dropout rates (0.2, 0.3, 0.5)
    - Learning rates (1e-3, 5e-4)
    
    Args:
        protein_name: Name of the protein to train on
        config: RNAConfig object
        device: Device to run on
        batch_size: Batch size for training
        num_epochs: Number of epochs per configuration
    
    Returns:
        best_config: Dictionary with best hyperparameters
        all_results: Dictionary with all results
    """
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SEARCH FOR {protein_name}")
    print(f"{'='*70}\n")
    
    # Define hyperparameter grid
    param_grid = {
        'num_filters': [32, 64, 128],
        'kernel_size': [6, 8, 10],
        'pooling_type': ['global_max', 'local_max', 'none'],
        'dropout_rate': [0.2, 0.3, 0.5],
        'learning_rate': [1e-3, 5e-4]
    }
    
    # Load data once - create loader instance that will be reused
    print("Loading data (this will take a minute or two)...")
    loader = RNACompeteLoader(config)  # Create loader once
    train_dataset = loader.get_data(protein_name, split='train')
    val_dataset = loader.get_data(protein_name, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Generate all combinations (this could be large!)
    # For practical purposes, we'll do a smaller random search
    print(f"\nNote: Full grid search would test {np.prod([len(v) for v in param_grid.values()])} combinations.")
    print("Performing smart sampling of hyperparameter space...\n")
    
    # Sample a subset of combinations
    all_results = []
    best_val_corr = -1.0
    best_config = None
    
    # Smart sampling: Test each key hyperparameter variation
    test_configs = [
        # Baseline
        {'num_filters': 64, 'kernel_size': 8, 'pooling_type': 'global_max', 
         'dropout_rate': 0.3, 'learning_rate': 1e-3},
        # Vary filter size
        {'num_filters': 32, 'kernel_size': 8, 'pooling_type': 'global_max', 
         'dropout_rate': 0.3, 'learning_rate': 1e-3},
        {'num_filters': 128, 'kernel_size': 8, 'pooling_type': 'global_max', 
         'dropout_rate': 0.3, 'learning_rate': 1e-3},
        # Vary kernel size
        {'num_filters': 64, 'kernel_size': 6, 'pooling_type': 'global_max', 
         'dropout_rate': 0.3, 'learning_rate': 1e-3},
        {'num_filters': 64, 'kernel_size': 10, 'pooling_type': 'global_max', 
         'dropout_rate': 0.3, 'learning_rate': 1e-3},
        # Vary pooling
        {'num_filters': 64, 'kernel_size': 8, 'pooling_type': 'local_max', 
         'dropout_rate': 0.3, 'learning_rate': 1e-3},
        {'num_filters': 64, 'kernel_size': 8, 'pooling_type': 'none', 
         'dropout_rate': 0.3, 'learning_rate': 1e-3},
        # Vary dropout
        {'num_filters': 64, 'kernel_size': 8, 'pooling_type': 'global_max', 
         'dropout_rate': 0.2, 'learning_rate': 1e-3},
        {'num_filters': 64, 'kernel_size': 8, 'pooling_type': 'global_max', 
         'dropout_rate': 0.5, 'learning_rate': 1e-3},
        # Vary learning rate
        {'num_filters': 64, 'kernel_size': 8, 'pooling_type': 'global_max', 
         'dropout_rate': 0.3, 'learning_rate': 5e-4},
    ]
    
    total_configs = len(test_configs)
    
    for idx, params in enumerate(test_configs):
        print(f"\n{'='*70}")
        print(f"Configuration {idx + 1}/{total_configs}")
        print(f"{'='*70}")
        print(f"Parameters: {params}")
        
        # Create model
        model = RNABindingCNN(
            num_filters=params['num_filters'],
            kernel_size=params['kernel_size'],
            pooling_type=params['pooling_type'],
            dropout_rate=params['dropout_rate']
        ).to(device)
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=params['learning_rate'],
            device=device,
            save_path=None  # Don't save intermediate models
        )
        
        # Get final validation correlation
        final_val_corr = history['val_corr'][-1]
        
        # Store results
        result = {
            'params': params,
            'final_val_corr': final_val_corr,
            'history': history
        }
        all_results.append(result)
        
        # Update best
        if final_val_corr > best_val_corr:
            best_val_corr = final_val_corr
            best_config = params.copy()
            print(f"\n>>> NEW BEST CONFIGURATION! Val Corr: {final_val_corr:.4f}")
    
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"\nBest Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest Validation Correlation: {best_val_corr:.4f}")
    print(f"{'='*70}\n")
    
    return best_config, all_results


# ============================================================================
# STEP 6: Data Exploration Only (summary + plots)
# ============================================================================

def run_data_exploration(
    protein_name: str,
    config: RNAConfig,
    output_dir: str = "results",
    splits: Tuple[str, ...] = ("train", "val", "test")
) -> Dict[str, Dict]:
    """Generate dataset summaries and plots without training."""
    os.makedirs(output_dir, exist_ok=True)
    loader = RNACompeteLoader(config)

    summaries: Dict[str, Dict] = {}
    for split in splits:
        print(f"Loading {protein_name} {split} split for exploration...")
        dataset = loader.get_data(protein_name, split=split)
        summary = dataset_summary(dataset)
        summaries[split] = summary

        summary_path = os.path.join(output_dir, f"{protein_name}_{split}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved summary to {summary_path}")

    return summaries


# ============================================================================
# STEP 7: Main Execution
# ============================================================================

def main():
    """
    Main execution function.
    
    This function:
    1. Sets up the environment (seed, device)
    2. Loads the data
    3. Performs hyperparameter search (optional)
    4. Trains the final model with best hyperparameters
    5. Evaluates on test set
    6. Saves results and plots
    """
    parser = argparse.ArgumentParser(description="RNAcompete CNN trainer and data explorer")
    parser.add_argument("--protein", default="RBFOX1", help="Protein name to process")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--hyperparam-search", action="store_true", help="Run hyperparameter search instead of default config")
    parser.add_argument("--explore-only", action="store_true", help="Only run data summary/plots and skip training")
    args = parser.parse_args()

    # ========================================================================
    # Configuration
    # ========================================================================

    configure_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = RNAConfig()
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Early exit: exploration-only mode
    if args.explore_only:
        print(f"\n{'='*70}")
        print(f"DATA EXPLORATION FOR {args.protein}")
        print(f"{'='*70}\n")
        run_data_exploration(protein_name=args.protein, config=config, output_dir="results")
        print("Data exploration finished. Skipping training as requested.")
        return

    # ========================================================================
    # Option 1: Hyperparameter Search (Time-consuming but thorough)
    # ========================================================================

    if args.hyperparam_search:
        best_config, all_results = hyperparameter_search(
            protein_name=args.protein,
            config=config,
            device=device,
            batch_size=args.batch_size,
            num_epochs=20  # Fewer epochs for search
        )

        results_file = f'results/{args.protein}_hyperparameter_search.json'
        with open(results_file, 'w') as f:
            serializable_results = []
            for result in all_results:
                serializable_results.append({
                    'params': result['params'],
                    'final_val_corr': result['final_val_corr']
                })
            json.dump({
                'best_config': best_config,
                'all_results': serializable_results
            }, f, indent=2)
        print(f"Hyperparameter search results saved to {results_file}")

    else:
        best_config = {
            'num_filters': 64,
            'kernel_size': 8,
            'pooling_type': 'global_max',
            'dropout_rate': 0.3,
            'learning_rate': 1e-3
        }
        print(f"\nUsing default configuration:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")

    # ========================================================================
    # Option 2: Train Final Model with Best/Default Configuration
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"TRAINING FINAL MODEL FOR {args.protein}")
    print(f"{'='*70}\n")

    print("Loading data (this will take a minute or two)...")
    loader = RNACompeteLoader(config)
    train_dataset = loader.get_data(args.protein, split='train')
    val_dataset = loader.get_data(args.protein, split='val')
    test_dataset = loader.get_data(args.protein, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set:   {len(val_dataset)} samples")
    print(f"Test set:  {len(test_dataset)} samples")

    model = RNABindingCNN(
        num_filters=best_config['num_filters'],
        kernel_size=best_config['kernel_size'],
        pooling_type=best_config['pooling_type'],
        dropout_rate=best_config['dropout_rate']
    ).to(device)

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    model_save_path = f"models/{args.protein}_best_model.pt"
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=best_config['learning_rate'],
        device=device,
        save_path=model_save_path
    )

    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION ON TEST SET")
    print(f"{'='*70}\n")

    checkpoint = torch.load(model_save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_corr = evaluate(model, test_loader, device)

    print(f"Test Loss:        {test_loss:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")

    final_results = {
        'protein': args.protein,
        'config': best_config,
        'test_loss': test_loss,
        'test_correlation': test_corr,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }

    results_file = f'results/{args.protein}_final_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    epochs = list(range(1, args.epochs + 1))

    plot(
        epochs=epochs,
        plottables={
            'Train Loss': history['train_loss'],
            'Val Loss': history['val_loss']
        },
        filename=f'results/{args.protein}_loss_curve.png'
    )
    print(f"Loss curve saved to results/{args.protein}_loss_curve.png")

    plot(
        epochs=epochs,
        plottables={
            'Train Correlation': history['train_corr'],
            'Val Correlation': history['val_corr']
        },
        filename=f'results/{args.protein}_correlation_curve.png',
        ylim=[0, 1]
    )
    print(f"Correlation curve saved to results/{args.protein}_correlation_curve.png")

    print(f"\n{'='*70}")
    print(f"ALL DONE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
