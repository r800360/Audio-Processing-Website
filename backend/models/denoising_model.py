# Example of a simple denoising model using librosa and numpy
import numpy as np
import librosa
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import pickle
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy.linalg import svd
import tensorflow as tf
from librosa import load
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class AudioDataset(Dataset):
    def __init__(self, audio_file, seq_len):
        self.audio, self.sr = librosa.load(audio_file, sr=None)
        self.audio = torch.tensor(self.audio, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.audio) - self.seq_len
    
    def __getitem__(self, idx):
        return self.audio[idx:idx+self.seq_len]

# Load the trained models
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


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        # Encode
        x, (hidden, cell) = self.encoder(x)
        # Decode
        x, _ = self.decoder(x, (hidden, cell))
        return x


def denoise_audio_autoencoder(filepath):
    input_dim = 1025
    hidden_dim = 256
    num_layers = 2
    seq_length = 400
    
    dataset = AudioDataset(filepath, seq_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
    num_epochs = 100

    for epoch in range(num_epochs):
        for x in dataset:
            x = torch.tensor(x, dtype=torch.float32)
            output = model(x)
            loss = criterion(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
    
    model.eval()
    denoised_audio = torch.zeros_like(dataset.audio)
    
    with torch.no_grad():
        for i in range(0, len(dataset.audio) - seq_length, seq_length):
            input_seq = dataset.audio[i:i+seq_length].unsqueeze(0)
            output_seq = model(input_seq)
            denoised_audio[i:i+seq_length] = output_seq.squeeze()
    
    output_path = filepath.replace('.wav', '_denoised.wav').replace('.mp3', '_denoised.mp3')
    sf.write(output_path, denoised_audio, dataset.sr)
    
    return output_path

def denoise_audio_lstm(filepath):
    input_dim = 64
    hidden_dim = 64
    num_layers = 2
    seq_length = 400
    
    dataset = AudioDataset(filepath, seq_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LSTMAutoencoder(input_dim, hidden_dim, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
    num_epochs = 100

    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = batch.unsqueeze(-1)  # Add channel dimension
            output = model(batch)
            loss = criterion(output, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        
    
    model.eval()
    denoised_audio = torch.zeros_like(dataset.audio)
    
    with torch.no_grad():
        for i in range(0, len(dataset.audio) - seq_length, seq_length):
            input_seq = dataset.audio[i:i+seq_length].unsqueeze(0).unsqueeze(-1)
            output_seq = model(input_seq)
            denoised_audio[i:i+seq_length] = output_seq.squeeze()
    
    output_path = filepath.replace('.wav', '_denoised.wav').replace('.mp3', '_denoised.mp3')
    sf.write(output_path, denoised_audio, dataset.sr)
    
    return output_path
    
# autoencoder = Autoencoder()
# autoencoder.load_state_dict(torch.load('autoencoder.pth'))
# autoencoder.eval()

# with open('gmm.pkl', 'rb') as f:
#     gmm = pickle.load(f)

# def denoise_audio(filepath):
#     y, sr = librosa.load(filepath, sr=None)
#     S = librosa.feature.melspectrogram(y, sr=sr)

#     # Denoise using the autoencoder
#     S_denoised = []
#     for frame in S.T:
#         frame_tensor = torch.tensor(frame, dtype=torch.float32)
#         frame_denoised = autoencoder(frame_tensor).detach().numpy()
#         S_denoised.append(frame_denoised)
#     S_denoised = np.array(S_denoised).T

#     # Further denoise using the GMM
#     S_denoised_flat = S_denoised.flatten().reshape(-1, 1)
#     gmm_labels = gmm.predict(S_denoised_flat)
#     S_denoised = S_denoised_flat[gmm_labels == 1].reshape(S_denoised.shape)

#     output_path = filepath.replace('.wav', '_denoised.wav')
#     librosa.output.write_wav(output_path, librosa.feature.inverse.mel_to_audio(S_denoised, sr=sr), sr)
#     return output_path


def denoise_audio(filepath):
    y, sr = librosa.load(filepath, sr=None)
    y_denoised = y#librosa.effects.remix(y)
    output_path = filepath.replace('.wav', '_denoised.wav').replace('.mp3', '_denoised.mp3')
    sf.write(output_path, y_denoised, sr)
    return output_path


