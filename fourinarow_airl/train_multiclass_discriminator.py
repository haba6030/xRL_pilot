"""
Multi-Class AIRL Discriminator - Step 0.3b

Goal: Extend binary discriminator to 4-class classification (h=1,2,3,4)

Benefits over binary:
- Better calibration (intermediate classes)
- Finer-grained h estimation
- Reduced bias from binary decision boundary

Usage:
    python3 train_multiclass_discriminator.py
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class MultiClassStateActionDataset(Dataset):
    """Dataset of (state, action) pairs with h labels (0=h1, 1=h2, 2=h3, 3=h4)"""

    def __init__(self, states, actions, labels):
        """
        Args:
            states: (N, 89) numpy array
            actions: (N,) numpy array of action indices
            labels: (N,) numpy array (0=h1, 1=h2, 2=h3, 3=h4)
        """
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
        self.labels = torch.LongTensor(labels)  # Long for CrossEntropyLoss

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.labels[idx]


class MultiClassDiscriminator(nn.Module):
    """
    Multi-class discriminator for h=1,2,3,4

    Architecture:
        Input: state (89-dim) + action (one-hot 36-dim) = 125-dim
        Hidden: [256, 128, 64]
        Output: 4 (logits for each h class)
    """

    def __init__(self, state_dim=89, action_dim=36, num_classes=4, hidden_dims=[256, 128, 64]):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_classes = num_classes

        # Build network
        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        # Output layer (logits for each class)
        layers.append(nn.Linear(input_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, state, action):
        """
        Args:
            state: (B, 89) tensor
            action: (B,) tensor of action indices

        Returns:
            logits: (B, num_classes) tensor
        """
        # One-hot encode actions
        action_onehot = torch.zeros(action.size(0), self.action_dim, device=state.device)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)

        # Concatenate state and action
        x = torch.cat([state, action_onehot], dim=1)

        # Forward pass
        logits = self.network(x)

        return logits


def load_trajectories(h, data_dir='data/separate_h_trajectories'):
    """Load trajectories for given h"""
    filepath = Path(data_dir) / f'trajectories_h{h}.pkl'
    if not filepath.exists():
        raise FileNotFoundError(f"Trajectories not found: {filepath}")
    with open(filepath, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


def trajectories_to_state_action_pairs(trajectories):
    """
    Convert trajectories to (state, action) pairs

    Args:
        trajectories: List of trajectory dicts with keys:
            - 'observations': (T+1, 89) array
            - 'actions': (T,) array

    Returns:
        states: (N, 89) array
        actions: (N,) array
    """
    states = []
    actions = []

    for traj in trajectories:
        obs = traj['observations']
        acts = traj['actions']

        # Use all (s_t, a_t) pairs except final state (no action)
        T = len(acts)
        states.append(obs[:T])  # s_0, ..., s_{T-1}
        actions.append(acts)     # a_0, ..., a_{T-1}

    states = np.vstack(states)
    actions = np.concatenate(actions)

    return states, actions


def prepare_dataset(h_values=[1, 2, 3, 4]):
    """Prepare training and test datasets for multi-class classification"""
    print("=" * 80)
    print("Loading Trajectories for Multi-Class Training")
    print("=" * 80)

    all_states = []
    all_actions = []
    all_labels = []

    for idx, h in enumerate(h_values):
        print(f"\nLoading h={h}...")

        try:
            trajs = load_trajectories(h=h)
            states, actions = trajectories_to_state_action_pairs(trajs)
            labels = np.full(len(states), idx)  # 0=h1, 1=h2, 2=h3, 3=h4

            all_states.append(states)
            all_actions.append(actions)
            all_labels.append(labels)

            print(f"  Trajectories: {len(trajs)}")
            print(f"  (state, action) pairs: {len(states)}")

        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")
            print(f"  Skipping h={h}")
            continue

    if len(all_states) == 0:
        raise ValueError("No trajectory data found! Generate trajectories first.")

    # Combine all data
    all_states = np.vstack(all_states)
    all_actions = np.concatenate(all_actions)
    all_labels = np.concatenate(all_labels)

    print(f"\nTotal pairs: {len(all_states)}")
    for idx, h in enumerate(h_values):
        count = np.sum(all_labels == idx)
        if count > 0:
            print(f"  h={h}: {count} ({count/len(all_states)*100:.1f}%)")

    # Train/test split (stratified to maintain class balance)
    train_states, test_states, train_actions, test_actions, train_labels, test_labels = \
        train_test_split(all_states, all_actions, all_labels,
                        test_size=0.2, random_state=42, stratify=all_labels)

    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(train_states)} pairs")
    print(f"  Test:  {len(test_states)} pairs")

    # Create datasets
    train_dataset = MultiClassStateActionDataset(train_states, train_actions, train_labels)
    test_dataset = MultiClassStateActionDataset(test_states, test_actions, test_labels)

    return train_dataset, test_dataset, h_values


def train_discriminator(model, train_loader, test_loader, num_epochs=50, lr=0.001):
    """Train the multi-class discriminator"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    print("\n" + "=" * 80)
    print("Training Multi-Class Discriminator")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {train_loader.batch_size}")

    best_test_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for states, actions, labels in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(states, actions)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * len(states)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(states)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for states, actions, labels in test_loader:
                states = states.to(device)
                actions = actions.to(device)
                labels = labels.to(device)

                logits = model(states, actions)
                loss = criterion(logits, labels)

                test_loss += loss.item() * len(states)
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += len(states)

        test_loss /= test_total
        test_acc = test_correct / test_total

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f} | "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.3f}")

    # Restore best model
    model.load_state_dict(best_model_state)

    print(f"\n✅ Best test accuracy: {best_test_acc:.3f}")

    return model, history, best_test_acc


def evaluate_discriminator(model, test_loader, h_values):
    """Detailed evaluation of multi-class discriminator"""
    device = next(model.parameters()).device
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for states, actions, labels in test_loader:
            states = states.to(device)
            actions = actions.to(device)

            logits = model(states, actions)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Overall accuracy
    accuracy = np.mean(all_labels == all_preds)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    print(f"\nOverall Accuracy: {accuracy:.3f}")

    print(f"\nConfusion Matrix:")
    print(f"         ", end="")
    for h in h_values:
        print(f"Pred h={h:2d}  ", end="")
    print()
    for i, h_true in enumerate(h_values):
        print(f"True h={h_true:2d}", end="")
        for j in range(len(h_values)):
            print(f"{cm[i, j]:8d}  ", end="")
        print()

    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    for i, h in enumerate(h_values):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  h={h}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs
    }


def visualize_results(history, eval_results, h_values):
    """Visualize training and evaluation results"""
    fig = plt.figure(figsize=(20, 10))

    # Plot 1: Training curves
    ax1 = plt.subplot(2, 4, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', alpha=0.7)
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Acc', alpha=0.7)
    ax2.axhline(y=0.25, color='gray', linestyle='--', label='Chance (25%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.0, 1.0])

    # Plot 3: Confusion Matrix
    ax3 = plt.subplot(2, 4, 3)
    cm = eval_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'h={h}' for h in h_values],
                yticklabels=[f'h={h}' for h in h_values],
                ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('Confusion Matrix')

    # Plot 4: Normalized Confusion Matrix
    ax4 = plt.subplot(2, 4, 4)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[f'h={h}' for h in h_values],
                yticklabels=[f'h={h}' for h in h_values],
                ax=ax4)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    ax4.set_title('Normalized Confusion Matrix')

    # Plot 5-8: Probability distributions for each class
    for plot_idx, h_idx in enumerate(range(len(h_values))):
        ax = plt.subplot(2, 4, 5 + plot_idx)

        # Get probabilities for true class h_idx
        mask = eval_results['labels'] == h_idx
        probs_for_class = eval_results['probabilities'][mask]

        if len(probs_for_class) > 0:
            # Plot histogram for each h
            for i, h in enumerate(h_values):
                ax.hist(probs_for_class[:, i], bins=20, alpha=0.5,
                       label=f'P(h={h})', density=True)

            ax.set_xlabel('Probability')
            ax.set_ylabel('Density')
            ax.set_title(f'Discriminator Output for True h={h_values[h_idx]}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path('figures') / 'multiclass_discriminator_results.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Multi-Class AIRL Discriminator Training")
    print("Goal: Classify h=1,2,3,4 from (state, action) pairs")
    print("=" * 80)

    # Check which h values have trajectories
    available_h = []
    for h in [1, 2, 3, 4]:
        filepath = Path('data/separate_h_trajectories') / f'trajectories_h{h}.pkl'
        if filepath.exists():
            available_h.append(h)
        else:
            print(f"⚠️  Missing trajectories for h={h}: {filepath}")

    if len(available_h) < 2:
        print("\n❌ ERROR: Need at least 2 h values with trajectories")
        print("Run generate_separate_h_trajectories.py first")
        return

    print(f"\n✅ Found trajectories for h={available_h}")

    # Prepare datasets
    train_dataset, test_dataset, h_values = prepare_dataset(h_values=available_h)

    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = MultiClassDiscriminator(
        state_dim=89,
        action_dim=36,
        num_classes=len(h_values),
        hidden_dims=[256, 128, 64]
    )

    print(f"\nModel Architecture:")
    print(f"  Input: state (89) + action_onehot (36) = 125")
    print(f"  Hidden: [256, 128, 64]")
    print(f"  Output: {len(h_values)} classes")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    model, history, best_acc = train_discriminator(
        model, train_loader, test_loader,
        num_epochs=50, lr=0.001
    )

    # Evaluate
    eval_results = evaluate_discriminator(model, test_loader, h_values)

    # Visualize
    visualize_results(history, eval_results, h_values)

    # Save model
    model_path = Path('models') / 'multiclass_discriminator.pt'
    model_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'h_values': h_values,
        'num_classes': len(h_values),
        'accuracy': best_acc,
        'eval_results': eval_results,
        'history': history
    }, model_path)
    print(f"\n✅ Saved model to {model_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nClasses: {h_values}")
    print(f"Test Accuracy: {best_acc:.3f}")
    print(f"Chance level: {1/len(h_values):.3f} ({100/len(h_values):.1f}%)")

    if best_acc > 1 / len(h_values) + 0.2:
        print(f"\n✅ SUCCESS: Discriminator significantly above chance")
        print(f"\nNext steps:")
        print(f"  → Re-estimate human players with multi-class discriminator")
        print(f"  → Analyze distribution of h across players")
        print(f"  → Compare with binary discriminator results")
    else:
        print(f"\n⚠️  Moderate performance: Need investigation")
        print(f"\nPossible actions:")
        print(f"  → Generate more trajectories")
        print(f"  → Try different architectures")
        print(f"  → Check for class imbalance")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
