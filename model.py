from unicodedata import bidirectional
from torch.autograd import Variable 
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math
device = torch.device("cuda")

class NaiveTransConv(nn.Module):    
	def __init__(self, numInputChannels, deep_channels, kernel=6):
		super(NaiveTransConv, self).__init__()
		self.convTrans1 = nn.ConvTranspose2d(in_channels=numInputChannels, out_channels=deep_channels, kernel_size=kernel, stride=2, padding = 2) #Ouutput=(Input Size - 1) * Strides + Filter Size - 2 * Padding + Ouput Padding
		self.convTrans2 = nn.ConvTranspose2d(in_channels=deep_channels, out_channels=numInputChannels, kernel_size=kernel, stride=2, padding = 2)
	def forward(self,x):
		x = self.convTrans1(x)
		x = self.convTrans2(x)
		return x
class ED_TransConv(nn.Module):
    def __init__(self,channels):
        super(ED_TransConv, self).__init__()
        # Input image size is (N, 3, H, W)
        #Encoder
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Upsample by a factor of 2
        #Decoder
        self.upsample1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        # Another convolutional layer
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        # Final upsample by a factor of 2 to achieve total 4x upsampling
        self.upsample2 = nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=2, stride=2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.upsample1(x)
        x = self.conv3(x)
        x = self.upsample2(x)
        return x

class ED_PixelShuffle(nn.Module):
    def __init__(self,channels,upscale_factor):
        super(ED_PixelShuffle, self).__init__()
        # Input image size is (N, 3, H, W)
        ps_channels = int(channels * (upscale_factor ** 2))
        #Enconder
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        #Decoder and Sub-pixel convolution layer
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.sub_pixel_conv = nn.Sequential(
            nn.Conv2d(128, ps_channels, 3, 1, 1),
            nn.PixelShuffle(upscale_factor))
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv4(x)
        x = self.sub_pixel_conv(x)
        return x
class VDSR(nn.Module):
	def __init__(self, numInputChannels, deep_channels, num_layers, scale_factor):
		# call the parent constructor
		super(VDSR, self).__init__()
		self.upsample = nn.Upsample(scale_factor=scale_factor,mode='bicubic')
		self.conv_first = nn.Conv2d(in_channels=numInputChannels, out_channels=deep_channels, kernel_size=3, padding=1)
            
		self.conv_middle = nn.ModuleList()
		for _ in range(num_layers-1):
			self.conv_middle.append(nn.Conv2d(in_channels=deep_channels, out_channels=deep_channels, kernel_size=3, padding=1)) 
		self.conv_middle = nn.Sequential(*self.conv_middle)
            
		self.conv_last = nn.Conv2d(in_channels=deep_channels, out_channels=numInputChannels, kernel_size=3, padding=1)

	def forward(self,x):
		x = self.upsample(x)
		x = F.relu(self.conv_first(x))
		for layer in self.conv_middle:
			x = F.relu(layer(x))
		x = self.conv_last(x)
		return x
class FSRCNN(nn.Module):
	def __init__(self, numInputChannels, d, s, m, scale_factor):
		# call the parent constructor
		super(FSRCNN, self).__init__()
		#Feature extraction (as in paper) Conv(filter,d,channels)
		self.feat_extraction = nn.Sequential(nn.Conv2d(in_channels=numInputChannels, out_channels=d, kernel_size=5, padding=5//2), 
                                       nn.PReLU(d))
            
		#Schrinking extraction (as in paper)
		self.schrinking = nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1), nn.PReLU(s))
		#Mapping (as in paper)
		self.mapping = nn.ModuleList()
		for _ in range(m):
			self.mapping.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=3//2))
			self.mapping.append(nn.PReLU(s))
		self.mapping = nn.Sequential(*self.mapping)
		#Expanding (as in paper)
		self.expanding = nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1), 
                                  nn.PReLU(d))
		#Deconvoltion
		self.deconv = nn.ConvTranspose2d(in_channels=d, out_channels=numInputChannels, kernel_size=9,
								   stride=scale_factor,padding=9//2, output_padding=scale_factor-1)
	    # Initialize model weights.
		self._initialize_weights()
	def forward(self,x):
		x = self.feat_extraction(x)
		x = self.schrinking(x)
		x = self.mapping(x)
		x = self.expanding(x)
		x = self.deconv(x)
		return x
	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
				nn.init.zeros_(module.bias.data)
		nn.init.normal_(self.deconv.weight.data, 0.0, 0.001)
		nn.init.zeros_(self.deconv.bias.data)

class SRCNN(nn.Module):
	#https://arxiv.org/abs/1501.00092v3
	def __init__(self, numInputChannels, d, s):
		super(SRCNN, self).__init__()
		self.upsample = nn.Upsample(scale_factor=4,mode='bicubic')
        # Feature extraction layer.
		self.feat_extraction = nn.Conv2d(in_channels=numInputChannels, out_channels=d, kernel_size=9,stride=1,padding=4)
        # Non-linear mapping layer.
		self.mapping = nn.Conv2d(in_channels=d, out_channels=s, kernel_size=5,stride=1,padding=2)
        # Rebuild the layer.
		self.reconstruction = nn.Conv2d(in_channels=s, out_channels=numInputChannels, kernel_size=5,stride=1,padding=2)
        # Initialize model weights.
		self._initialize_weights()
	def forward(self, x):
		x = self.upsample(x)
		x = F.relu(self.feat_extraction(x))
		x = F.relu(self.mapping(x))
		x = self.reconstruction(x)
		return x
	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
				nn.init.zeros_(module.bias.data)
		nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
		nn.init.zeros_(self.reconstruction.bias.data)


class ESPCN(nn.Module):
    #https://arxiv.org/pdf/1609.05158.pdf
    def __init__(self,in_channels: int,out_channels: int,
                 channels: int,upscale_factor: int):
        super(ESPCN, self).__init__()
        self.hidden_channels = channels // 2
        out_channels = int(out_channels * (upscale_factor ** 2))

        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, 5, 1, 2),
            nn.Tanh(),
            nn.Conv2d(channels, self.hidden_channels, 3, 1, 1),
            nn.Tanh(),
        )
        # Sub-pixel convolution layer
        self.sub_pixel_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels, out_channels, 3, 1, 1),
            nn.PixelShuffle(upscale_factor))
        self._initialize_weights()
    def forward(self, x):
        x = self.feature_maps(x)
        x = self.sub_pixel_conv(x)
        return x
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == self.hidden_channels:
                    nn.init.normal_(module.weight.data,0.0,0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data,0.0,math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                    nn.init.zeros_(module.bias.data)
					