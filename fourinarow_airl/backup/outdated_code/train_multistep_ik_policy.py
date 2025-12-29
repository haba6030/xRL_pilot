"""
Train Multi-Step Inverse Kinematics Policies

Based on Mhammedi et al. (2023) Algorithm 2 (IKDP).

For each h ∈ {1, 2, 3, 4}, train a policy that:
- Takes current state and h-step future state as input
- Predicts the action that was taken at current state

Objective: max_{f,φ} Σ log f((a,h) | φ(x_t), φ(x_{t+h}))

where:
- f: policy network
- φ: state encoder
- a: action taken at t
- h: planning depth

After training, we can generate trajectories with each policy
and compare their action distributions.

Usage:
    python3 train_multistep_ik_policy.py --h 1
    python3 train_multistep_ik_policy.py --h 2
    python3 train_multistep_ik_policy.py --h 3
    python3 train_multistep_ik_policy.py --h 4
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


class MultiStepIKDataset(Dataset):
    """Dataset for multi-step IK pairs"""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        state_current = torch.FloatTensor(pair['state_current'])
        state_future = torch.FloatTensor(pair['state_future'])
        action = torch.LongTensor([pair['action']])[0]
        h = torch.LongTensor([pair['h']])[0]

        return state_current, state_future, action, h


class MultiStepIKPolicy(nn.Module):
    """
    Multi-Step Inverse Kinematics Policy Network.

    Architecture:
    1. State encoder φ: maps 90-dim state to latent embedding
    2. Policy head f: maps (φ(x_t), φ(x_{t+h}), h) to action logits
    """

    def __init__(self, state_dim=90, embedding_dim=128, n_actions=36, max_h=4):
        super().__init__()

        # State encoder (shared for current and future states)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.ReLU()
        )

        # h embedding (learnable embedding for planning depth)
        self.h_embedding = nn.Embedding(max_h + 1, embedding_dim)

        # Policy head (predicts action from encoded states + h)
        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),  # z_current + z_future + h_embed
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_actions)
        )

    def forward(self, state_current, state_future, h):
        """
        Args:
            state_current: (batch, 90) current state
            state_future: (batch, 90) future state (h steps ahead)
            h: (batch,) planning depth

        Returns:
            action_logits: (batch, 36) unnormalized action scores
        """
        # Encode states
        z_current = self.state_encoder(state_current)  # (batch, embedding_dim)
        z_future = self.state_encoder(state_future)    # (batch, embedding_dim)

        # Embed h
        h_embed = self.h_embedding(h)  # (batch, embedding_dim)

        # Concatenate features
        features = torch.cat([z_current, z_future, h_embed], dim=1)  # (batch, 3*embedding_dim)

        # Predict action logits
        action_logits = self.policy_head(features)  # (batch, 36)

        return action_logits

    def predict_action(self, state_current, state_future, h, temperature=1.0):
        """
        Predict action with optional temperature scaling.

        Args:
            state_current: (90,) numpy array
            state_future: (90,) numpy array
            h: int
            temperature: float, higher = more random

        Returns:
            action: int (0-35)
            probs: (36,) numpy array of action probabilities
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors
            state_current = torch.FloatTensor(state_current).unsqueeze(0)  # (1, 90)
            state_future = torch.FloatTensor(state_future).unsqueeze(0)
            h_tensor = torch.LongTensor([h])

            # Get logits
            logits = self.forward(state_current, state_future, h_tensor)  # (1, 36)

            # Apply temperature
            logits = logits / temperature

            # Softmax to get probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (36,)

            # Sample action
            action = np.random.choice(36, p=probs)

            return action, probs


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for state_curr, state_fut, actions, h in dataloader:
        state_curr = state_curr.to(device)
        state_fut = state_fut.to(device)
        actions = actions.to(device)
        h = h.to(device)

        # Forward pass
        logits = model(state_curr, state_fut, h)
        loss = criterion(logits, actions)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == actions).sum().item()
        total += actions.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for state_curr, state_fut, actions, h in dataloader:
            state_curr = state_curr.to(device)
            state_fut = state_fut.to(device)
            actions = actions.to(device)
            h = h.to(device)

            logits = model(state_curr, state_fut, h)
            loss = criterion(logits, actions)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == actions).sum().item()
            total += actions.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Planning depth')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    args = parser.parse_args()

    print("=" * 80)
    print(f"Training Multi-Step IK Policy for h={args.h}")
    print("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print(f"\nLoading data for h={args.h}...")
    data_path = Path('data/multistep_ik') / f'ik_pairs_h{args.h}.pkl'
    with open(data_path, 'rb') as f:
        pairs = pickle.load(f)

    print(f"Total pairs: {len(pairs)}")

    # Train/val split
    n_val = int(len(pairs) * args.val_split)
    n_train = len(pairs) - n_val

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(pairs))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")

    # Create datasets
    train_dataset = MultiStepIKDataset(train_pairs)
    val_dataset = MultiStepIKDataset(val_pairs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print(f"\nCreating model...")
    model = MultiStepIKPolicy(
        state_dim=90,
        embedding_dim=128,
        n_actions=36,
        max_h=4
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.3f}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.3f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Save model
    output_dir = Path('models/multistep_ik')
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = output_dir / f'policy_h{args.h}.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'h': args.h,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_loss': best_val_loss,
    }, model_path)

    print(f"\n✅ Saved model to {model_path}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Final val acc: {val_accs[-1]:.3f}")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss (h={args.h})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs, label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Training Accuracy (h={args.h})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = Path('figures') / f'training_h{args.h}.png'
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved training curves to {fig_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
