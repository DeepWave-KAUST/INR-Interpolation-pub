import torch
from torch import nn
import numpy as np
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

class Fourier_reparam_linear(nn.Module):
    """Linear layer whose weights are reparameterized over a fixed Fourier basis.

    The effective weight matrix is reconstructed as ``lamb @ bases`` where ``bases`` is a
    fixed bank of cosine functions and ``lamb`` holds the learnable coefficients.
    """
    def __init__(self,in_features,out_features,high_freq_num,low_freq_num,phi_num,alpha):
        """Build the basis bank and learnable coefficients.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            high_freq_num (int): Number of high-frequency cosine bases.
            low_freq_num (int): Number of low-frequency cosine bases.
            phi_num (int): Number of phase offsets per frequency.
            alpha (float): Scaling factor applied to the basis bank.
        """
        super(Fourier_reparam_linear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num =high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha=alpha
        self.bases=self.init_bases()
        self.lamb=self.init_lamb()
        self.bias=nn.Parameter(torch.Tensor(self.out_features,1),requires_grad=True)
        self.init_bias()

    def init_bases(self):
        """Build the fixed (non-trainable) bank of cosine basis functions."""
        phi_set=np.array([2*math.pi*i/self.phi_num for i in range(self.phi_num)])
        high_freq=np.array([i+1 for i in range(self.high_freq_num)])
        low_freq=np.array([(i+1)/self.low_freq_num for i in range(self.low_freq_num)])
        if len(low_freq)!=0:
            T_max=2*math.pi/low_freq[0]
        else:
            T_max=2*math.pi/min(high_freq)
        points=np.linspace(-T_max/2,T_max/2,self.in_features)
        bases=torch.Tensor((self.high_freq_num+self.low_freq_num)*self.phi_num,self.in_features)
        i=0
        for freq in low_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        for freq in high_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        bases=self.alpha*bases
        bases=nn.Parameter(bases,requires_grad=False)
        return bases


    def init_lamb(self):
        """Initialize the learnable Fourier coefficients."""
        self.lamb=torch.Tensor(self.out_features,(self.high_freq_num+self.low_freq_num)*self.phi_num)
        with torch.no_grad():
            m=(self.low_freq_num+self.high_freq_num)*self.phi_num
            for i in range(m):
                dominator=torch.norm(self.bases[i,:],p=2)
                self.lamb[:,i]=nn.init.uniform_(self.lamb[:,i],-np.sqrt(6/m)/dominator,np.sqrt(6/m)/dominator)
        self.lamb=nn.Parameter(self.lamb,requires_grad=True)
        return self.lamb

    def init_bias(self):
        """Initialize the bias to zero."""
        with torch.no_grad():
            nn.init.zeros_(self.bias)

    def forward(self,x):
        """Apply the reparameterized linear transform to ``x``."""
        weight=torch.matmul(self.lamb,self.bases)
        output=torch.matmul(x,weight.transpose(0,1))
        output=output+self.bias.T
        return output

class SineLayer(nn.Module):
    """SIREN sine layer (Sitzmann et al. 2020): ``sin(omega_0 * (Wx + b))``."""
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        """Build the linear layer and initialize its weights.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Whether to include a bias term.
            is_first (bool): Whether this is the first layer (affects initialization).
            omega_0 (float): Frequency multiplier of the sine activation.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        """Initialize weights following the SIREN initialization scheme."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        """Apply the linear layer followed by the scaled sine activation."""
        return torch.sin(self.omega_0 * self.linear(input))

class sin_fr_layer(nn.Module):
    """SIREN-style sine layer with Fourier-reparameterized weights.

    Combines the Fourier weight reparameterization of :class:`Fourier_reparam_linear`
    with a SIREN sine activation ``sin(omega_0 * output)``.
    """
    def __init__(self, in_features, out_features, high_freq_num,low_freq_num,phi_num,alpha,omega_0=30.0):
        """Build the basis bank, learnable coefficients and bias.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            high_freq_num (int): Number of high-frequency cosine bases.
            low_freq_num (int): Number of low-frequency cosine bases.
            phi_num (int): Number of phase offsets per frequency.
            alpha (float): Scaling factor applied to the basis bank.
            omega_0 (float): Frequency multiplier of the sine activation.
        """
        super().__init__()
        super(sin_fr_layer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num =high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha=alpha
        self.omega_0=omega_0
        self.bases=self.init_bases()
        self.lamb=self.init_lamb()
        self.bias=nn.Parameter(torch.Tensor(self.out_features,1),requires_grad=True)
        self.init_bias()

    def init_bases(self):
        """Build the fixed (non-trainable) bank of cosine basis functions."""
        phi_set=np.array([2*math.pi*i/self.phi_num for i in range(self.phi_num)])
        high_freq=np.array([i+1 for i in range(self.high_freq_num)])
        low_freq=np.array([(i+1)/self.low_freq_num for i in range(self.low_freq_num)])
        if len(low_freq)!=0:
            T_max=2*math.pi/low_freq[0]
        else:
            T_max=2*math.pi/min(high_freq)
        points=np.linspace(-T_max/2,T_max/2,self.in_features)
        bases=torch.Tensor((self.high_freq_num+self.low_freq_num)*self.phi_num,self.in_features)
        i=0
        for freq in low_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        for freq in high_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        bases=self.alpha*bases
        bases=nn.Parameter(bases,requires_grad=False)
        return bases


    def init_lamb(self):
        """Initialize the learnable Fourier coefficients (scaled by ``omega_0``)."""
        self.lamb=torch.Tensor(self.out_features,(self.high_freq_num+self.low_freq_num)*self.phi_num)
        with torch.no_grad():
            m=(self.low_freq_num+self.high_freq_num)*self.phi_num
            for i in range(m):
                dominator=torch.norm(self.bases[i,:],p=2)
                self.lamb[:,i]=nn.init.uniform_(self.lamb[:,i],-np.sqrt(6/m)/dominator/self.omega_0,np.sqrt(6/m)/dominator/self.omega_0)
        self.lamb=nn.Parameter(self.lamb,requires_grad=True)
        return self.lamb

    def init_bias(self):
        """Initialize the bias to zero."""
        with torch.no_grad():
            nn.init.zeros_(self.bias)

    def forward(self,x):
        """Apply the reparameterized linear transform followed by the sine activation."""
        weight=torch.matmul(self.lamb,self.bases)
        output=torch.matmul(x,weight.transpose(0,1))
        output=output+self.bias.T
        return torch.sin(self.omega_0*output)

class INR(nn.Module):
    """Configurable implicit neural representation.

    Supports several backbones selected by ``mode``: ``relu``, ``relu+fr``, ``relu+pe``,
    ``sin`` (SIREN) and ``sin+fr`` (SIREN with Fourier-reparameterized layers), with an
    optional NeRF positional encoding on the input.
    """
    def __init__(self,mode,in_features,hidden_features,hidden_layers,out_features,outermost_linear,high_freq_num,low_freq_num,
    phi_num,alpha,first_omega_0,hidden_omega_0,pe):
        """Build the network for the requested ``mode``.

        Args:
            mode (str): Backbone type: 'relu', 'relu+fr', 'relu+pe', 'sin' or 'sin+fr'.
            in_features (int): Number of input features.
            hidden_features (int): Number of features per hidden layer.
            hidden_layers (int): Number of hidden layers.
            out_features (int): Number of output features.
            outermost_linear (bool): If True, the last layer is a plain linear layer.
            high_freq_num (int): Number of high-frequency Fourier bases (for '+fr' modes).
            low_freq_num (int): Number of low-frequency Fourier bases (for '+fr' modes).
            phi_num (int): Number of phase offsets per frequency (for '+fr' modes).
            alpha (float): Scaling factor for the Fourier bases.
            first_omega_0 (float): Sine frequency of the first layer.
            hidden_omega_0 (float): Sine frequency of the hidden layers.
            pe (bool): If True, apply NeRF positional encoding to the input.
        """
        super().__init__()
        self.net=[]
        self.pe=pe
        if pe==True:
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,sidelength=256,fn_samples=None,use_nyquist=True)
            in_features=self.positional_encoding.out_dim
        self.net=[]
        if mode=='relu':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            for i in range(hidden_layers):
                self.net.append(nn.Linear(hidden_features,hidden_features))
                self.net.append(nn.ReLU())

        if mode=='relu+fr':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            for i in range(hidden_layers):
                self.net.append(Fourier_reparam_linear(hidden_features,hidden_features,high_freq_num,low_freq_num,phi_num,alpha))
                self.net.append(nn.ReLU())

        if mode=='relu+pe':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            for i in range(hidden_layers):
                self.net.append(nn.Linear(hidden_features,hidden_features))
                self.net.append(nn.ReLU())

        if mode=='sin':
            self.net.append(SineLayer(in_features, hidden_features,is_first=True, omega_0=first_omega_0))
            for i in range(hidden_layers):
                self.net.append(SineLayer(hidden_features, hidden_features,is_first=False, omega_0=hidden_omega_0))

        if mode=='sin+fr':
            self.net.append(SineLayer(in_features, hidden_features,is_first=True, omega_0=first_omega_0))
            for i in range(hidden_layers):
                self.net.append(sin_fr_layer(hidden_features,hidden_features,high_freq_num,low_freq_num,phi_num,alpha,hidden_omega_0))
        if outermost_linear==True:
            final_linear = nn.Linear(hidden_features, out_features)
            if mode =='sin+fr' or mode=='sin':
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6/ hidden_features)/hidden_omega_0,np.sqrt(6 / hidden_features)/hidden_omega_0)
            else:
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6/ hidden_features),np.sqrt(6 / hidden_features))
            self.net.append(final_linear)
        else:
            if mode=='relu' or mode=='relu+fr':
                final_linear=nn.Linear(hidden_features,out_features)
                self.net.append(final_linear)
                self.net.append(nn.ReLU())
            if mode=='sin' or mode=='sin+fr':
                self.net.append(SineLayer(hidden_features, out_features,is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """Run a forward pass.

        Returns:
            tuple: ``(output, x)`` where ``x`` is the (optionally encoded) input.
        """
        if self.pe==True:
            x=self.positional_encoding(x)
            print(x.shape)
        output=self.net(x)
        return output, x

def get_INR(mode,in_features, hidden_features, hidden_layers,
            out_features, outermost_linear, high_freq_num,low_freq_num,phi_num,alpha,first_omega_0,
            hidden_omega_0, pe):
    '''
        Function to get a class instance for a given type of
        implicit neural representation

        Inputs:
            mode: non-linear activation functions and Fourier reparameterized training
            in_features: Number of input features. 2 for image, 3 for volume and so on.
            hidden_features: Number of features per hidden layer
            hidden_layers: Number of hidden layers
            out_features; Number of outputs features. 3 for color image, 1 for grayscale or volume and so on
            outermost_linear (True): If True, do not apply nonlin
                just before output
            high_freq_num: The number of the high frequence in the Fourier bases B.
            low_freq_num: The number of the low frequence in the Fourier bases B.
            phi_num: The number of the phase in the Fourier bases B.
            (high_freq_num, low_freq_num, phi_num): The detailed description can be found in Sec. 3.2 of our paper
            alpha: The role can be found in Appendix A. Detailed proof of Theorem 2 of our paper. Empirically, alpha=0.05 for relu;
                alpha=0.01 for sin and so on.
            first_omega0 (30): For siren: Omega for first layer
            hidden_omega0 (30): For siren and siren+fr: Omega for hidden layers
            pos_encode (False): If True apply positional encoding
        Output: An INR class instance
    '''
    model=INR(mode, in_features, hidden_features, hidden_layers, out_features, outermost_linear, high_freq_num, low_freq_num, phi_num, alpha, first_omega_0, hidden_omega_0, pe)

    return model
