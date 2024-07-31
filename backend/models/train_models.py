import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1025, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1025),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(data):
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 50

    for epoch in range(epochs):
        for x in data:
            x = torch.tensor(x, dtype=torch.float32)
            output = model(x)
            loss = criterion(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), 'autoencoder.pth')
    return model

def train_gmm(data):
    gmm = GaussianMixture(n_components=2, covariance_type='full', max_iter=100)
    gmm.fit(data)
    return gmm

def load_data(filepaths):
    data = []
    for filepath in filepaths:
        y, sr = librosa.load(filepath, sr=None)
        S = librosa.feature.melspectrogram(y, sr=sr)
        data.append(S.T)
    data = np.concatenate(data, axis=0)
    return data

if __name__ == "__main__":
    # Replace with your dataset paths
    filepaths = ['audio1.wav', 'audio2.wav', 'audio3.wav']
    data = load_data(filepaths)

    # Train and save the autoencoder model
    autoencoder = train_autoencoder(data)
    torch.save(autoencoder.state_dict(), 'autoencoder.pth')

    # Train and save the GMM model
    gmm = train_gmm(data)
    with open('gmm.pkl', 'wb') as f:
        pickle.dump(gmm, f)
