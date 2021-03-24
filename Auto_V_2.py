import os
import torch as torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd

data_PDF = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/strus/all_PDFs.csv")

Dataset =data_PDF.loc[:, data_PDF.columns != 'true']
X_train, X_test, y_train, y_test = train_test_split(data_PDF.loc[:, data_PDF.columns != 'true'], data_PDF.iloc[:,-1:], test_size=0.20, random_state=42)

NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

x = torch.from_numpy(X_train.values).type(torch.FloatTensor)
y = torch.from_numpy(X_test.values).type(torch.FloatTensor)


"""
class AETrainingData(Dataset):
    '''
        Format the training dataset to be input into the auto encoder.
        Takes in dataframe and converts it to a PyTorch Tensor
    '''

    def __init__(self, x_train):
        self.x = x_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        '''
            Returns a example from the data set as a pytorch tensor.
        '''
        # Get example/target pair at idx as numpy arrays
        x, y = self.x.iloc[idx].values, self.x.iloc[idx].values

        # Convert to torch tensor
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.FloatTensor)

        # Return pair        
        return {'input': x, 'target': y}
"""


class Encoder(nn.Module):
    def __init__(self, input_shape, drop_prob=0):
        super(Encoder, self).__init__()
        self.drop_prob = drop_prob

        self.e1 = nn.Linear(input_shape, 2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.e2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.e3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)

        self.e4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.e5 = nn.Linear(256, 50)

    def forward(self, input):
        block1 = F.dropout(self.bn1(F.elu(self.e1(input))), p=self.drop_prob)
        block2 = F.dropout(self.bn2(F.elu(self.e2(block1))), p=self.drop_prob)
        block3 = F.dropout(self.bn3(F.elu(self.e3(block2))), p=self.drop_prob)
        block4 = F.dropout(self.bn4(F.elu(self.e4(block3))), p=self.drop_prob)
        encoded_representation = F.tanh(self.e5(block4))
        return encoded_representation
net = Encoder((x.shape))

class Decoder(nn.Module):
    def __init__(self, output_shape, drop_prob=0):
        super(Decoder, self).__init__()
        self.drop_prob = drop_prob

        self.d = nn.Linear(50, 256)
        self.bn = nn.BatchNorm1d(256)

        self.d1 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.d2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.d3 = nn.Linear(1024, 2048)
        self.bn3 = nn.BatchNorm1d(2048)

        self.d4 = nn.Linear(2048, output_shape)

    def forward(self, input):
        block = F.dropout(self.bn(F.elu(self.d(input))), p=self.drop_prob)
        block1 = F.dropout(self.bn1(F.elu(self.d1(block))), p=self.drop_prob)
        block2 = F.dropout(self.bn2(F.elu(self.d2(block1))), p=self.drop_prob)
        block3 = F.dropout(self.bn3(F.elu(self.d3(block2))), p=self.drop_prob)
        reconstruction = F.sigmoid(self.d4(block3))
        return reconstruction


def train_ae(input_tensor, target_tensor, encoder, decoder,
             encoder_optimizer, decoder_optimizer, criterion):
    # clear the gradients in the optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Forward pass through

    encoded_representation = encoder(input_tensor)
    reconstruction = decoder(encoded_representation)

    # Compute the loss
    loss = criterion(reconstruction, target_tensor)

    # Compute the gradients
    loss.backward()

    # Step the optimizers to update the model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Return the loss value to track training progress
    return loss.item()

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def trainIters(encoder, decoder, dataloader, epochs, print_every_n_batches=100, learning_rate=0.001):
    # keep track of losses
    plot_losses = []

    # Initialize Encoder Optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-8)

    # Initialize Decoder Optimizer
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=1e-8)

    # Specify loss function
    criterion = nn.MSELoss(reduce=True)

    # Cycle through epochs
    for epoch in range(epochs):

        print(f'Epoch {epoch + 1}/{epochs}')
        # Cycle through batches
        for i, batch in enumerate(dataloader):

            input_tensor = batch['input'].to(device)
            target_tensor = batch['target'].to(device)

            loss = train_ae(input_tensor, target_tensor, encoder, decoder,
                            encoder_optimizer, decoder_optimizer, criterion)

            if i % print_every_n_batches == 0 and i != 0:
                print(loss)
                plot_losses.append(loss)
    return plot_losses

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


train_ae(x,y,Encoder,Decoder,optimizer,optimizer,criterion)