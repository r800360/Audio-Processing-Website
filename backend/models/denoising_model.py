# Example of a simple denoising model using librosa and numpy
import numpy as np
import librosa
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import pickle

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

# autoencoder = Autoencoder()
# autoencoder.load_state_dict(torch.load('autoencoder.pth'))
# autoencoder.eval()

# with open('gmm.pkl', 'rb') as f:
#     gmm = pickle.load(f)

def denoise_audio(filepath):
    y, sr = librosa.load(filepath, sr=None)
    S = librosa.feature.melspectrogram(y, sr=sr)

    # Denoise using the autoencoder
    S_denoised = []
    for frame in S.T:
        frame_tensor = torch.tensor(frame, dtype=torch.float32)
        frame_denoised = autoencoder(frame_tensor).detach().numpy()
        S_denoised.append(frame_denoised)
    S_denoised = np.array(S_denoised).T

    # Further denoise using the GMM
    S_denoised_flat = S_denoised.flatten().reshape(-1, 1)
    gmm_labels = gmm.predict(S_denoised_flat)
    S_denoised = S_denoised_flat[gmm_labels == 1].reshape(S_denoised.shape)

    output_path = filepath.replace('.wav', '_denoised.wav')
    librosa.output.write_wav(output_path, librosa.feature.inverse.mel_to_audio(S_denoised, sr=sr), sr)
    return output_path


def denoise_audio(filepath):
    y, sr = librosa.load(filepath, sr=None)
    y_denoised = librosa.effects.remix(y)
    output_path = filepath.replace('.wav', '_denoised.wav')
    librosa.output.write_wav(output_path, y_denoised, sr)
    return output_path
