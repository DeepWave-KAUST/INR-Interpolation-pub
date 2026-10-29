import torch
import numpy as np
import torch.nn as nn
import tinycudann as tc
import math

class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        """Configure the encoding for a given input dimensionality.

        Args:
            in_features (int): Number of input coordinate dimensions (1, 2 or 3).
            sidelength (int or tuple, optional): Image side length, required for 2D inputs.
            fn_samples (int, optional): Number of samples, required for 1D inputs.
            use_nyquist (bool): If True, pick the number of frequencies from the Nyquist rate.
        """
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        """Return the number of frequency bands supported by the Nyquist rate."""
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        """Encode coordinates into sinusoidal positional features.

        Args:
            coords (Tensor): Input coordinates of shape ``(batch, ..., in_features)``.

        Returns:
            Tensor: Encoded coordinates of shape ``(batch, -1, out_dim)``.
        """
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

class SineMLP(nn.Module):
    """SIREN-style MLP with an optional hash-grid input encoding."""
    def __init__(self,
                 layers: int = 4,
                 features: int = 64,
                 out_dim: int = 1,
                 in_dim: int = 2,
                 omega: float = 30,
                 use_bias=True,
                 use_hash=False,
                 hash_config=None):
        """Build the SIREN layers and initialize their weights.

        Args:
            layers (int): Number of hidden layers.
            features (int): Number of features per hidden layer.
            out_dim (int): Number of output features.
            in_dim (int): Number of input features.
            omega (float): Frequency multiplier of the sine activations.
            use_bias (bool): Whether linear layers include a bias term.
            use_hash (bool): If True, encode inputs with a tiny-cuda-nn hash grid.
            hash_config (dict): Hash-grid configuration passed to ``tinycudann``.
        """
        super().__init__()
        self.use_hash = use_hash

        if use_hash:
            self.enc = tc.Encoding(in_dim, hash_config, dtype=torch.float32)
            in_dim = self.enc.n_output_dims
            print('Reset in_dim to', in_dim)


        self.first_layer = nn.Linear(in_dim,
                                     features,
                                     bias=use_bias)
        self.last_layer = nn.Linear(features,
                                    out_dim,
                                    bias=use_bias)
        self.omega = omega
        # initialize the weights
        with torch.no_grad():
            self.first_layer.weight.uniform_(-1 / in_dim,
                                             1 / in_dim)
            self.last_layer.weight.uniform_(-np.sqrt(6./features)/omega,
                                            np.sqrt(6./features)/omega)

        self.layers = []
        for _ in range(layers):
            layer = nn.Linear(features, features, bias=use_bias)
            with torch.no_grad():
                layer.weight.uniform_(-np.sqrt(6./features)/omega,
                                      np.sqrt(6./features)/omega)
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, omega=None):
        """Run a forward pass.

        Args:
            x (Tensor): Input coordinates.
            omega (float, optional): Overrides the default sine frequency.

        Returns:
            Tensor: The output, or ``(output, hash_enc)`` when a hash grid is used.
        """
        shape = x.shape[:-1]
        omega = omega if omega is not None else self.omega
        if self.use_hash:
            hash_enc = self.enc(x)
            x = torch.sin(omega * self.first_layer(hash_enc))
        else:
            x = torch.sin(omega * self.first_layer(x))
        for layer in self.layers:
            x = torch.sin(omega * layer(x))
        x = self.last_layer(x)
        if self.use_hash:
            return x, hash_enc
        else:
            return x

class HybridEncodingMLP(nn.Module):
    """SIREN MLP fed by a weighted concatenation of hash-grid and Fourier encodings.

    The hash and Fourier encodings are blended with weights ``w_hash`` and ``1 - w_hash``
    supplied at forward time, allowing a schedule from Fourier- to hash-dominated features.
    """
    def __init__(self,
                 layers=4,
                 features=64,
                 out_dim=1,
                 in_dim=2,
                 omega=30,
                 use_hash=True,
                 hash_config=None,
                 use_fourier=True,
                 use_bias=True,
                 ):
        """Build the encoders and the SIREN MLP.

        Args:
            layers (int): Number of hidden layers.
            features (int): Number of features per hidden layer.
            out_dim (int): Number of output features.
            in_dim (int): Number of input features.
            omega (float): Frequency multiplier of the sine activations.
            use_hash (bool): If True, include a tiny-cuda-nn hash-grid encoding.
            hash_config (dict): Hash-grid configuration passed to ``tinycudann``.
            use_fourier (bool): If True, include a NeRF positional encoding.
            use_bias (bool): Whether the hidden linear layers include a bias term.
        """
        super().__init__()

        self.use_hash = use_hash
        self.use_fourier = use_fourier
        self.omega = omega

        # -------- 1) Hash encoding --------
        if use_hash:
            self.hash_enc = tc.Encoding(in_dim, hash_config, dtype=torch.float32)
            hash_dim = self.hash_enc.n_output_dims
        else:
            hash_dim = 0

        # -------- 2) Fourier encoding --------
        if use_fourier:
            self.fouier_enc = PosEncodingNeRF(in_features=in_dim, sidelength=256, fn_samples=None, use_nyquist=True)
            fouier_dim=self.fouier_enc.out_dim
        else:
            fouier_dim = 0

        total_in_dim = hash_dim + fouier_dim
        print("Final input dim =", total_in_dim)

        self.first_layer = nn.Linear(total_in_dim, features)
        self.last_layer = nn.Linear(features, out_dim)

        with torch.no_grad():
            self.first_layer.weight.uniform_(-1/total_in_dim, 1/total_in_dim)
            self.last_layer.weight.uniform_(-np.sqrt(6./features)/omega,
                                            np.sqrt(6./features)/omega)

        self.layers = []
        for _ in range(layers):
            layer = nn.Linear(features, features, bias=use_bias)
            with torch.no_grad():
                layer.weight.uniform_(-np.sqrt(6./features)/omega,
                                      np.sqrt(6./features)/omega)
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, w_hash, mode='hybrid'):
        """Encode, blend and decode the input coordinates.

        Args:
            x (Tensor): Input coordinates.
            w_hash (float): Blend weight for the hash encoding (Fourier gets ``1 - w_hash``).
            mode (str): 'hybrid' keeps both feature groups, which is Eq. (17) and the
                configuration every result of the paper is trained in. 'fourier_only'
                and 'hash_only' zero the hash and the Fourier group respectively, which
                is the channel-zeroing probe of Fig. 5. Note that these two settings are
                an inference-time diagnostic applied to a model trained in 'hybrid'
                mode; the independently trained single-branch models of Table 2 are
                obtained instead by building the network with ``use_hash=False`` or
                ``use_fourier=False``.

        Returns:
            tuple: ``(out, h_enc, f_enc)`` when both encodings are active, else ``(out, None, None)``.
        """
        if mode not in ('hybrid', 'fourier_only', 'hash_only'):
            raise ValueError(
                "mode must be 'hybrid', 'fourier_only' or 'hash_only', got %r" % (mode,))

        parts = []
        w_ff = 1.0 - w_hash

        if self.use_hash:
            h_enc = self.hash_enc(x)
            parts.append(torch.zeros_like(h_enc) if mode == 'fourier_only'
                         else w_hash * h_enc)

        if self.use_fourier:
            f_enc = self.fouier_enc(x)[:, 0, :]
            parts.append(torch.zeros_like(f_enc) if mode == 'hash_only'
                         else w_ff * f_enc)

        x_enc = torch.cat(parts, dim=-1)

        # ----- Standard SIREN MLP -----
        x = torch.sin(self.omega * self.first_layer(x_enc))
        for layer in self.layers:
            x = torch.sin(self.omega * layer(x))
        out = self.last_layer(x)

        if self.use_hash and self.use_fourier:
            return out, h_enc, f_enc
        else:
            return out, None, None


class HybridFusionBlock(nn.Module):
    """
    Stable fusion block for Hash + Fourier encodings.
    Uses gated feature fusion instead of multiplication.
    """

    def __init__(self, in_dim, hidden_dim):
        """Build the projection, gate and normalization layers.

        Args:
            in_dim (int): Number of input features.
            hidden_dim (int): Number of output features.
        """
        super().__init__()

        self.linear = nn.Linear(in_dim, hidden_dim)
        self.gate = nn.Linear(in_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x):
        """Apply gated fusion ``norm(linear(x) * sigmoid(gate(x)))``."""

        feat = self.linear(x)
        gate = torch.sigmoid(self.gate(x))

        out = feat * gate
        out = self.norm(out)

        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over a bank of (log-spaced) frequency bands."""
    def __init__(self, in_channels=2, N_freqs=64, logscale=True):
        """Configure the frequency bands.

        Args:
            in_channels (int): Number of input channels.
            N_freqs (int): Number of frequency bands.
            logscale (bool): If True, space frequencies logarithmically, else linearly.
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** N_freqs, N_freqs)

    def forward(self, x):
        """Concatenate the input with its sine/cosine features at every frequency band."""
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, dim=-1)


class LegendreEncoder(nn.Module):
    """Encode coordinates with Legendre polynomials up to a given order."""
    def __init__(self, in_features: int = 2, order: int = 128):
        """Configure the encoder.

        Args:
            in_features (int): Number of input features.
            order (int): Number of Legendre polynomials per input feature.
        """
        super().__init__()
        self.in_features = in_features
        self.order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Legendre polynomials on ``x`` (clamped to ``[-1, 1]``).

        Args:
            x (Tensor): Input of shape ``(B, N, in_features)``.

        Returns:
            Tensor: Encoded features of shape ``(B, N, in_features * order)``.
        """
        x = torch.clamp(x, -1.0, 1.0)
        B, N, D = x.shape
        assert D == self.in_features

        out = []
        for d in range(D):
            xd = x[..., d].unsqueeze(-1)  # [B, N, 1]
            P = [torch.ones_like(xd), xd]  # P0, P1

            for n in range(2, self.order):
                pn = ((2 * n - 1) * xd * P[-1] - (n - 1) * P[-2]) / n
                P.append(pn)

            P_all = torch.cat(P[:self.order], dim=-1)  # [B, N, order]
            out.append(P_all)

        return torch.cat(out, dim=-1)  # [B, N, D * order]


class ChebyshevEncoder(nn.Module):
    """Encode coordinates with Chebyshev polynomials up to a given order."""
    def __init__(self, in_features: int = 2, order: int = 30):
        """Configure the encoder.

        Args:
            in_features (int): Number of input features.
            order (int): Number of Chebyshev polynomials per input feature.
        """
        super().__init__()
        self.in_features = in_features
        self.order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Chebyshev polynomials on ``x`` via the recurrence relation.

        Args:
            x (Tensor): Input of shape ``(B, N, in_features)``.

        Returns:
            Tensor: Encoded features of shape ``(B, N, in_features * order)``.
        """
        B, N, D = x.shape
        assert D == self.in_features
        out = []
        for d in range(D):
            xd = x[..., d].unsqueeze(-1)  # [B, N, 1]
            T = [torch.ones_like(xd), xd]  # T0, T1
            for _ in range(2, self.order):
                Tn = 2 * xd * T[-1] - T[-2]
                T.append(Tn)
            T_all = torch.cat(T[:self.order], dim=-1)  # [B, N, order]
            out.append(T_all)
        return torch.cat(out, dim=-1)  # [B, N, D * order]


class RandomFourierEncoder(nn.Module):
    """Random Fourier feature encoding with a fixed (non-trainable) projection."""
    def __init__(self, in_features: int = 2, mapping_size: int = 88, scale: float = 30.0):
        """Sample and freeze the random projection matrix.

        Args:
            in_features (int): Number of input features.
            mapping_size (int): Number of random frequencies.
            scale (float): Standard deviation of the sampled frequencies.
        """
        super().__init__()
        self.B = nn.Parameter(torch.randn(mapping_size, in_features) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``x`` to ``[sin(2*pi*xB), cos(2*pi*xB)]``.

        Args:
            x (Tensor): Input of shape ``(B, N, in_features)``.

        Returns:
            Tensor: Encoded features of shape ``(B, N, 2 * mapping_size)``.
        """
        x_proj = torch.matmul(x, self.B.t())  # [B, N, mapping_size]
        sin_feat = torch.sin(2 * np.pi * x_proj)
        cos_feat = torch.cos(2 * np.pi * x_proj)
        return torch.cat([sin_feat, cos_feat], dim=-1)  # [B, N, 2 * mapping_size]


class CAFEBlock(nn.Module):
    """Multiplicative fusion block: the element-wise product of several linear branches."""

    def __init__(self, in_features: int, hidden_features: int, num_branches: int = 3):
        """Build the parallel linear branches.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of output features per branch.
            num_branches (int): Number of branches multiplied together.
        """
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            nn.Linear(in_features, hidden_features, bias=True)
            for _ in range(num_branches)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the element-wise product of all branch outputs."""
        out = self.branches[0](x)
        for i in range(1, self.num_branches):
            out = out * self.branches[i](x)
        return out


class BackboneMLP(nn.Module):
    """Plain ReLU MLP backbone (an identity map when ``hidden_layers == 0``)."""

    def __init__(self, hidden_features: int, hidden_layers: int = 1):
        """Build the backbone.

        Args:
            hidden_features (int): Width of each hidden layer.
            hidden_layers (int): Number of linear+ReLU blocks.
        """
        super().__init__()
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features, bias=True))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers) if hidden_layers > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone over ``x``."""
        return self.net(x)


class CAFE_Net(nn.Module):
    """Network combining random Fourier and Chebyshev encodings via a CAFE fusion block."""

    def __init__(self,
                 in_features: int = 2,
                 out_features: int = 1,
                 rff_mapping_size: int = 96,
                 rff_scale: float = 10.0,
                 cheb_order: int = 32,
                 hidden_features: int = 256,
                 num_branches: int = 2,
                 hidden_layers: int = 2):
        """Build the encoders, fusion block, backbone and output layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            rff_mapping_size (int): Number of random Fourier frequencies.
            rff_scale (float): Scale of the random Fourier frequencies.
            cheb_order (int): Chebyshev polynomial order.
            hidden_features (int): Backbone width (defaults to the encoded dim when 0).
            num_branches (int): Number of branches in the CAFE fusion block.
            hidden_layers (int): Number of backbone hidden layers.
        """
        super().__init__()
        self.encoder_rff = RandomFourierEncoder(in_features, rff_mapping_size, rff_scale)
        self.encoder_cheb = ChebyshevEncoder(in_features, cheb_order)

        encoded_dim = (rff_mapping_size * 2) + (cheb_order * in_features)

        if hidden_features == 0:
            hidden_features = encoded_dim
        self.mult_block = CAFEBlock(encoded_dim, hidden_features, num_branches)
        self.backbone = BackboneMLP(hidden_features, hidden_layers)
        self.out_linear = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, fuse, run the backbone and project to the output."""
        x_rff = self.encoder_rff(x)
        x_cheb = self.encoder_cheb(x)
        x_encoded = torch.cat([x_rff, x_cheb], dim=-1)

        feat = self.mult_block(x_encoded)
        feat = self.backbone(feat)
        out = self.out_linear(feat)

        return out

class CAFE_Net_our(nn.Module):
    """CAFE network variant that replaces random Fourier features with a hash-grid encoding."""

    def __init__(self,
                 in_features: int = 2,
                 out_features: int = 1,
                 rff_mapping_size: int = 96,
                 rff_scale: float = 10.0,
                 cheb_order: int = 32,
                 hidden_features: int = 256,
                 num_branches: int = 2,
                 hidden_layers: int = 2,
                 hash_config={}):
        """Build the hash and Chebyshev encoders, fusion block, backbone and output layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            rff_mapping_size (int): Unused; kept for signature compatibility with ``CAFE_Net``.
            rff_scale (float): Unused; kept for signature compatibility with ``CAFE_Net``.
            cheb_order (int): Chebyshev polynomial order.
            hidden_features (int): Backbone width (defaults to the encoded dim when 0).
            num_branches (int): Number of branches in the CAFE fusion block.
            hidden_layers (int): Number of backbone hidden layers.
            hash_config (dict): Hash-grid configuration passed to ``tinycudann``.
        """
        super().__init__()
        self.encoder_cheb = ChebyshevEncoder(in_features, cheb_order)
        self.hash_enc = tc.Encoding(in_features, hash_config, dtype=torch.float32)
        hash_dim = self.hash_enc.n_output_dims

        encoded_dim = hash_dim + (cheb_order * in_features)
        if hidden_features == 0:
            hidden_features = encoded_dim
        self.mult_block = CAFEBlock(encoded_dim, hidden_features, num_branches)
        self.backbone = BackboneMLP(hidden_features, hidden_layers)
        self.out_linear = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, w_hash) -> torch.Tensor:
        """Encode, blend hash/Chebyshev features by ``w_hash``, fuse and project to the output.

        Args:
            x (Tensor): Input coordinates of shape ``(B, N, in_features)``.
            w_hash (float): Blend weight for the hash encoding (Chebyshev gets ``1 - w_hash``).

        Returns:
            Tensor: The network output.
        """
        x_cheb = self.encoder_cheb(x)
        hash_input = x.reshape(-1, 2)
        x_hash = self.hash_enc(hash_input)
        x_hash = x_hash.view(x.shape[0], x.shape[1], -1)

        x_encoded = torch.cat([x_hash*w_hash, x_cheb*(1-w_hash)], dim=-1)

        feat = self.mult_block(x_encoded)
        feat = self.backbone(feat)
        out = self.out_linear(feat)

        return out

class PE_CAFE_Net(nn.Module):
    """CAFE network using a log-spaced positional encoding alongside Chebyshev features."""

    def __init__(self,
                 in_features: int = 2,
                 out_features: int = 3,
                 N_freqs: int = 10,
                 cheb_order: int = 20,
                 hidden_features: int = 0,
                 num_branches: int = 2,
                 hidden_layers: int = 1):
        """Build the encoders, fusion block, backbone and output layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            N_freqs (int): Number of positional-encoding frequency bands.
            cheb_order (int): Chebyshev polynomial order.
            hidden_features (int): Backbone width (defaults to the encoded dim when 0).
            num_branches (int): Number of branches in the CAFE fusion block.
            hidden_layers (int): Number of backbone hidden layers.
        """
        super().__init__()
        self.encoder_rff = PositionalEncoding(in_features, N_freqs= N_freqs)
        self.encoder_cheb = ChebyshevEncoder(in_features, cheb_order)

        encoded_dim = (N_freqs * 2 * in_features) + (cheb_order * in_features)
        if hidden_features == 0:
            hidden_features = encoded_dim
        self.mult_block = CAFEBlock(encoded_dim, hidden_features, num_branches)
        self.backbone = BackboneMLP(hidden_features, hidden_layers)
        self.out_linear = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, fuse, run the backbone and project to the output."""
        x_rff = self.encoder_rff(x)
        x_cheb = self.encoder_cheb(x)
        x_encoded = torch.cat([x_rff, x_cheb], dim=-1)

        feat = self.mult_block(x_encoded)
        feat = self.backbone(feat)
        out = self.out_linear(feat)

        return out
