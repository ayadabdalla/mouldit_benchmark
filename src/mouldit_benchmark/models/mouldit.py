import torch
import torch.nn as nn

from .architecture import Architecture


class MouldIt(nn.Module):
    """
    The MouldIt variational autoencoder (paper: MouldIt Model): an encoder to the mean and log variance
    of a diagonal Gaussian over `code` dimensions, a decoder, and one sigmoid head per class.
    """

    def __init__(self, architecture: Architecture, code: int, n_classes: int):
        super().__init__()
        self.encoder_layers = architecture.build_encoder()
        self.decoder_layers = architecture.build_decoder()
        hidden = architecture.hidden_dim
        self.fc_mu = nn.Linear(hidden, code)
        self.fc_mu.apply(_init_weights)
        self.fc_logvar = nn.Linear(hidden, code)
        self.fc_logvar.apply(_init_weights)
        self.classification_layer = nn.Linear(code, n_classes)
        self.classification_layer.apply(_init_weights)
        # Machine setting parameter (MSP) head: never trained or used.  Creating it keeps the random
        # initialisation of the published runs, and lets their saved weights load.
        self.msp_layer = nn.Linear(code, 1)
        self.msp_layer.apply(_init_weights)
        for layer in [*self.encoder_layers, *self.decoder_layers]:
            layer.apply(_init_weights)

    @classmethod
    def from_config(cls, architecture: str, seq_len: int, n_channels: int, code: int, n_classes: int):
        return cls(Architecture.from_yaml(architecture, seq_len, n_channels, code), code, n_classes)

    def encode(self, x):
        """(mu, logvar), each (B, code); logvar clamped to [-10, 10]."""
        h = self.encoder_layers(x)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -10, 10)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z, shape):
        return self.decoder_layers(z).view(shape)

    def classify(self, z):
        """Class probabilities, one-vs-rest."""
        return torch.sigmoid(self.classification_layer(z))

    def forward(self, x):
        """(reconstruction, class probabilities, mu, logvar); z is sampled in training and mu at inference."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(z, x.shape), self.classify(z), mu, logvar


def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
