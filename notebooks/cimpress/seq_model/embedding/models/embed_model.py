"""
    Design of the Cimpress autoencoder:
    1. Input size: this version fix the input to 100 * 100. If input images are smaller than
    2. use maxpooling ?:
    3. How to export embedding ?:
    4. Embedding size ?:
"""

import torch as th
import torch.nn as nn


class ConvEncoderDecoder(nn.Module):
    """
    Will use simple Conv2D and Relu layers to build this Coder without MaxPooling.
    1. Input size = N, 3, 100, 100;
    2. Embedding size = 576
    """
    def __init__(self):
        super(ConvEncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),       # N, 16, 50, 50 d=40,000
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),      # N, 32, 25, 25 d=20,000
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),                 # N, 64, 12, 12 d=9,216
            nn.ReLU()
        )

        self.embedder = nn.Linear(9216, 576)
        self.debedder = nn.Linear(576, 9216)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),                                # N, 32, 25, 25 d=20,000
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # N, 16, 50, 50 d=40,000
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # N, 3, 100, 100 d=30,000
            nn.Sigmoid()
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.encoder(x)
        x = th.flatten(x, start_dim=1, end_dim=-1)
        x = self.embedder(x)
        x = self.debedder(x)
        x = x.view(-1, 64, 12, 12)
        x = self.decoder(x)
        return x
