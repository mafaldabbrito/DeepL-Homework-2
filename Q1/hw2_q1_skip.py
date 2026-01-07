import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from medmnist import INFO

# ================= USER CONFIGURATION =================
MAXPOOL = False
SOFTMAX = False

if MAXPOOL:
    OUTPUT_DIR = "./results_q1_2"
    if SOFTMAX:
        CHECKPOINT_FILENAME = "maxpool_softmax_checkpoint.pth"
    else:
        CHECKPOINT_FILENAME = "maxpool_logits_checkpoint.pth"
else:
    OUTPUT_DIR = "./results_q1_1"
    if SOFTMAX:
        CHECKPOINT_FILENAME = "nomaxpool_softmax_checkpoint.pth"
    else:
        CHECKPOINT_FILENAME = "nomaxpool_logits_checkpoint.pth"
# ======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
info = INFO['bloodmnist']
n_classes = len(info['label'])

class Net(nn.Module):
    def __init__(self, use_softmax=False):
        super(Net, self).__init__()
        self.use_softmax = use_softmax
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256) 
        self.fc2 = nn.Linear(256, n_classes)     
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.use_softmax:
            x = self.softmax(x)
        return x

class NetMaxPool(nn.Module):
    def __init__(self, use_softmax=False):
        super(NetMaxPool, self).__init__()
        self.use_softmax = use_softmax
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256) 
        self.fc2 = nn.Linear(256, n_classes) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.use_softmax:
            x = self.softmax(x)
        return x

def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig(f'{name}.png', bbox_inches='tight')
    print(f"Saved plot: {name}.png")

def main():
    if not os.path.exists(CHECKPOINT_FILENAME):
        print(f"Error: File '{CHECKPOINT_FILENAME}' not found!")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading {CHECKPOINT_FILENAME}...")
    checkpoint = torch.load(CHECKPOINT_FILENAME, map_location=device)

    loaded_train_losses = checkpoint.get('train_losses', [])
    loaded_val_accs = checkpoint.get('val_accs', [])
    
    best_val = checkpoint.get('best_val_acc', 'N/A')
    best_test = checkpoint.get('best_test_acc', 'N/A')

    print(f"\n--- Metrics Found in Checkpoint ---")
    print(f"Epochs trained: {len(loaded_train_losses)}")
    print(f"Best Validation Accuracy: {best_val}")
    print(f"Test Acc at Best Val: {best_test}")

    config = CHECKPOINT_FILENAME.replace('_checkpoint.pth', '')
    epoch_range = range(1, len(loaded_train_losses) + 1)
    
    plot(epoch_range, loaded_train_losses, ylabel='Loss', name=f'{OUTPUT_DIR}/CNN-training-loss-{config}')
    plot(epoch_range, loaded_val_accs, ylabel='Accuracy', name=f'{OUTPUT_DIR}/CNN-validation-accuracy-{config}')

    # load model if needed
    # if MAXPOOL:
    #     model = NetMaxPool(use_softmax=SOFTMAX).to(device)
    # else:
    #     model = Net(use_softmax=SOFTMAX).to(device)
    
    # try:
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     print("\nModel weights loaded successfully into architecture.")
    # except Exception as e:
    #     print(f"\nWarning: Could not load weights into model. \nReason: {e}")

if __name__ == '__main__':
    main()