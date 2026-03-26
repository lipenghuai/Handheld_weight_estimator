# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.optim import Adam
from sklearn.decomposition import PCA
from torch.optim import Adam

from PointCloudProcessor import PointCloudProcessor
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PointNet Encoder
class PointNetEncoder(nn.Module):
    def __init__(self, output_size=128):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=False)[0]
        return x

# PointNet Decoder
class PointNetDecoder(nn.Module):
    def __init__(self, num_points=2048, latent_size=128):
        super(PointNetDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(latent_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_points * 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.num_points, 3)

# PointNet Autoencoder
class PointNetAutoencoder(nn.Module):
    def __init__(self, num_points=500, latent_size=128):
        super(PointNetAutoencoder, self).__init__()
        self.encoder = PointNetEncoder(output_size=latent_size)
        self.decoder = PointNetDecoder(num_points=num_points, latent_size=latent_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # PointNet expects [batch, channels, points]
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
