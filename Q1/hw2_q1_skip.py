import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST, INFO
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np



device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self, use_softmax=False):
        super(Net, self).__init__()
        self.use_softmax = use_softmax
        
        # 1. Conv Layer: 3 input channels -> 32 output channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        
        # 2. Conv Layer: 32 -> 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 3. Conv Layer: 64 -> 128 output channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size: 128 channels * 28 width * 28 height = 100,352
        self.fc1 = nn.Linear(128 * 28 * 28, 256) # 
        self.fc2 = nn.Linear(256, n_classes)     # 
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Apply Conv -> ReLU for each block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the tensor for the linear layer
        x = torch.flatten(x, 1)
        
        # Linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Return logits by default (required for CrossEntropyLoss)
        if self.use_softmax:
            x = self.softmax(x)
            
        return x
    
def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.png' % (name), bbox_inches='tight')

# 1. Load the checkpoint
checkpoint = torch.load("q1_1_bloodmnist_checkpoint.pth")

# 2. Extract the history lists
loaded_train_losses = checkpoint['train_losses']
loaded_val_accs = checkpoint['val_accs']
loaded_test_accs = checkpoint['test_accs']

# 3. Plot immediately using the loaded data
config = "lr0.001_adam_nomaxpool_nosoftmax"
epoch_range = range(1, len(loaded_train_losses) + 1)
plot(epoch_range, loaded_train_losses, ylabel='Loss', name='./results_q1_1/CNN-training-loss-{}'.format(config))
plot(epoch_range, loaded_val_accs, ylabel='Accuracy', name='./results_q1_1/CNN-validation-accuracy-{}'.format(config))
plot(epoch_range, loaded_test_accs, ylabel='Accuracy', name='./results_q1_1/CNN-test-accuracy-{}'.format(config))

# 4. (Optional) If you also want to use the model for prediction
# model = Net(use_softmax=False).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])