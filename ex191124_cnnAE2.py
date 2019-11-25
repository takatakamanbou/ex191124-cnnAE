import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):

    def __init__(self, in_shape, C1, C2, C3, H):
        super(AE, self).__init__()

        X = torch.rand((1,) + in_shape)
        print('# input:', X.shape)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=X.shape[1], out_channels=C1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        X = self.conv1(X)
        print('# conv1 output:', X.shape)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv1[0].out_channels, out_channels=C2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        X = self.conv2(X)
        print('# conv2 output:', X.shape)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2[0].out_channels, out_channels=C3, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        X = self.conv3(X)
        print('# conv3 output:', X.shape)
        
        self.conv3_shape = np.array(X.shape[1:])
        print('# conv3_shape:', self.conv3_shape)

        X = X.view((-1, np.prod(self.conv3_shape)))
        print('# flatten output:', X.shape)

        self.fc1 = nn.Sequential(
            nn.Linear(X.shape[1], H),
            nn.LeakyReLU(),
        )
        X = self.fc1(X)
        print('# fc1 output:', X.shape)

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1[0].out_features, self.fc1[0].in_features),
            nn.LeakyReLU()
        )
        X = self.fc2(X)
        print('# fc2 output:', X.shape)

        X = X.view((-1,) + tuple(self.conv3_shape))
        print('# reshaped output:', X.shape)

        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.conv3_shape[0], C2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        X = self.tconv1(X)
        print('# tconv1 output:', X.shape)

        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.tconv1[0].out_channels, C1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        X = self.tconv2(X)
        print('# tconv2 output:', X.shape)


        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.tconv2[0].out_channels, in_shape[0], kernel_size=4, stride=2, padding=1, bias=False)
        )
        X = self.tconv3(X)
        print('# tconv3 output:', X.shape)


    def encoder(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = X.view((-1, np.prod(self.conv3_shape)))
        X = self.fc1(X)
        return X

    def decoder(self, X):
        X = self.fc2(X)
        X = X.view((-1,) + tuple(self.conv3_shape))
        X = self.tconv1(X)
        X = self.tconv2(X)
        X = self.tconv3(X)
        return X
        

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X


def evaluate(model, X, Y, bindex):

    nbatch = bindex.shape[0]
    sqe = 0.0
    with torch.no_grad():
        for ib in range(nbatch):
            ii = np.where(bindex[ib, :])[0]
            output = model(X[ii, ::])
            #print('@', output.shape)
            sqe += F.mse_loss(output, Y[ii, ::], reduction='sum').item()
            #print('@@', sqe)

    return sqe / X.shape[0]