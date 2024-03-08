import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


NUM_EPOCHS = 20

class ElectricityPricePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_layer1 = nn.Linear(input_size, 10)        
        self.hidden_layer2 = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)

        self.tansig = nn.Tanh()

    def forward(self, x):
        x = self.tansig(self.hidden_layer1(x))
        x = self.tansig(self.hidden_layer2(x))
        return self.output(x)


def train_model(model, train_loader):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    loss_list = []

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epoch", unit="epoch"):
        # Train set
        epoch_loss_list = []
        for X, y in train_loader:
            preds = model(X)
            loss = loss_function(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_list.append( loss.item() )
        
        loss_list.append( np.mean(epoch_loss_list) )
    
    return loss_list


def get_model_predictions(model, test_loader):
    y_pred = []

    # Assesses the model on the test set
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            preds = model(X)       
            y_pred.extend(preds.cpu().numpy())
    
    return y_pred