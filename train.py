import os
import sys

from torch.utils.data import DataLoader
from torchvision import models

from dataload import *
import numpy as np
from torch import nn
import torch.optim as optim
import tqdm
import warnings
import matplotlib.pyplot as plt

dir = r"./animals/animals" # 데이터셋 경로 알아서 수정
warnings.filterwarnings(action='ignore') # Jupter 경고 무시

train_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((224, 224)),
                                      ])

train_dataset = CustomDataset(dir, train_transform)
valid_dataset = CustomDataset(dir, train_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_dataset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

class_list = os.listdir('./animals/animals')

def resnet50():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 90)
    return model

device_txt = 'cuda:0'
device = torch.device(device_txt)
model = resnet50().to(device)
# model.load_state_dict(torch.load(r'D:\Cup_noodle\hbj\5_0.08881767552982016.pth'))

def train(model, train_loader, train_dataset, optimizer):
    model.train()

    train_running_loss = 0.0
    train_running_correct = 0

    pbar = tqdm.tqdm(train_loader, unit='batch')
    for img, label in pbar:
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        outputs = model(img)
        # print(outputs.data)
        # print(outputs.data.shape)

        loss = criterion(outputs, label)
        train_running_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        # print('preds', preds)
        # print('labels:', label)

        train_running_correct += (preds == label).sum().item()

        loss.backward()
        optimizer.step()

    _train_loss = train_running_loss / (len(train_dataset) / batch_size)
    _train_accuracy = 100. * train_running_correct / len(train_dataset)

    return _train_loss, _train_accuracy

def validate():
    pass

epochs = 20
lr = 0.001
best_loss = 1000
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    for epoch in range(epochs):
        print(f'-------epoch {epoch + 1}')

        phase = ['train', 'valid'] if (epoch + 1) / 10 == 0 else ['train']

        train_loss, train_accuracy = train(model, train_loader, train_dataset, optimizer)

        if best_loss > train_loss:
            best_model = model.state_dict()
            best_loss = train_loss

        print('train accuracy:', train_accuracy)
        print('train loss:', train_loss)

    torch.save(best_model, f'./{epochs}_{best_loss}.pth')
