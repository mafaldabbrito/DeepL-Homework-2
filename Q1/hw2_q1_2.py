# -*- coding: utf-8 -*-


#https://github.com/MedMNIST/MedMNIST


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from medmnist import BloodMNIST, INFO

import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

import time
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Data Loading
data_flag = 'bloodmnist'
print(data_flag)
info = INFO[data_flag]
print(len(info['label']))
n_classes = len(info['label'])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

total_start = time.time()

class NetMaxPool(nn.Module):
    def __init__(self, use_softmax=False):
        super(NetMaxPool, self).__init__()
        self.use_softmax = use_softmax
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256) 
        self.fc2 = nn.Linear(256, 8)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Conditional Softmax
        if self.use_softmax:
            x = self.softmax(x)
        
        return x

def train_epoch(loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)
        
        optimizer.zero_grad()

        outputs = model(imgs)

        loss = criterion(outputs, labels)

        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(loader, model):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.tolist()

    return accuracy_score(targets, preds)


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.png' % (name), bbox_inches='tight')


def main_q1_2(use_softmax):
    epochs = 200
    batch_size = 64
    lr = 0.001

    train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
    val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
    test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = NetMaxPool(use_softmax).to(device) 

    optimizer = optim.Adam(model.parameters(), lr=lr)
 
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []
    best_val_acc = 0.0
    best_model_test_acc = 0.0

    for epoch in range(epochs):

        epoch_start = time.time()

        train_loss = train_epoch(train_loader, model, criterion, optimizer)
        val_acc = evaluate(val_loader, model)

        train_losses.append(train_loss)
        val_accs.append(val_acc)

        test_acc = evaluate(test_loader, model)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_test_acc = test_acc

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        print(f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.2f} sec")


    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_test_acc': best_model_test_acc,
        'epoch': epochs
    }

    # save model
    filename_suffix = "softmax" if use_softmax else "logits"
    torch.save(checkpoint, f"maxpool_{filename_suffix}_checkpoint.pth")
    print(f"Saved checkpoint for {filename_suffix} version.")

    print(f"\nRESULTS ({filename_suffix}):")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy at Best Val: {best_model_test_acc:.4f}")


    total_end = time.time()
    total_time = total_end - total_start

    print(f"\nTotal training time: {total_time/60:.2f} minutes "
        f"({total_time:.2f} seconds)")

    results_dir = './results_q1_2'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    if use_softmax:
        config = "maxpool_softmax"
    else:
        config = "maxpool_logits"

    epoch_range = range(1, epochs + 1)

    plot(epoch_range, train_losses, ylabel='Loss', name=f'{results_dir}/CNN-training-loss-{config}')
    plot(epoch_range, val_accs, ylabel='Accuracy', name=f'{results_dir}/CNN-validation-accuracy-{config}')

if __name__ == '__main__':
    main_q1_2(False)