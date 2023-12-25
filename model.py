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
	def __init__(self, numInputChannels, deep_channels, kernel=2):
		super(NaiveTransConv, self).__init__()
		self.convTrans1 = nn.ConvTranspose2d(in_channels=numInputChannels, out_channels=deep_channels, kernel_size=kernel, stride=kernel)
		self.convTrans2 = nn.ConvTranspose2d(in_channels=deep_channels, out_channels=numInputChannels, kernel_size=kernel, stride=kernel)
	def forward(self,x):
		x = self.convTrans1(x)
		x = self.convTrans2(x)
		return x
class ED_UpsamplingModel(nn.Module):
    def __init__(self,channels):
        super(ED_UpsamplingModel, self).__init__()
        self.norm = torch.nn.BatchNorm2d(num_features=channels)
        # Input image size is (N, 3, H, W)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Upsample by a factor of 2
        self.upsample1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        # Another convolutional layer
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        # Final upsample by a factor of 2 to achieve total 4x upsampling
        self.upsample2 = nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=2, stride=2)
    def forward(self, x):
        #x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.upsample1(x)
        x = self.conv3(x)
        x = self.upsample2(x)
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
		self.deconv = nn.ConvTranspose2d(in_channels=d, out_channels=numInputChannels, kernel_size=9,stride=scale_factor,padding=9//2, output_padding=scale_factor-1)
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
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            upscale_factor: int):
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
					

class ConvLSTM_cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=False):
        super(ConvLSTM_cell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels = input_dim + self.hidden_dim, out_channels = 4 * self.hidden_dim,
                        kernel_size = kernel_size, padding = padding,bias=bias)

    def forward(self, x, prev_states):
        h_prev, c_prev = prev_states

        # Concatenate along the channel axis
        combined = torch.cat([x, h_prev], dim=1)
        combined_conv = self.conv(combined)

        # Split the tensor into four parts for each gate
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Input, forget, output, and cell gates
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Compute the next cell and hidden states
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
        
    def init_hidden(self, batch_size, height, width):
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device = self.conv.weight.device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device = self.conv.weight.device)
        return h, c    

class ConvLSTM_custom(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, n_layers, bias = True):
        super(ConvLSTM_custom, self).__init__()
  
        self.n_layers = n_layers
    
        self.cell_list = nn.ModuleList()
        for i in range(n_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.cell_list.append(ConvLSTM_cell(cur_input_dim, hidden_dim, kernel_size, bias))
        #Last layer
        self.cell_list = nn.Sequential(*self.cell_list)

        self.final_layer = nn.Conv2d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=3, padding=1) #ConvLSTM_cell(cur_input_dim, hidden_dim, kernel_size, bias)#

    def forward(self, x): #Batch first
        batch_size, seq_len, num_channels, height, width = x.size()

        # Initialize hidden states
        init_hidden = [self.cell_list[i].init_hidden(batch_size, height, width) for i in range(self.n_layers)]

        cur_layer_input = x
        layer_output_list = []
        layer_state_list = []

        for i in range(self.n_layers):
            h, c = init_hidden[i]
            output_sequence = []

            for t in range(seq_len):
                h, c = self.cell_list[i](cur_layer_input[:, t, :, :, :], (h, c))
                output_sequence.append(h)

            stacked_outputs = torch.stack(output_sequence, dim=1)
            cur_layer_input = stacked_outputs

            layer_output_list.append(stacked_outputs)
            layer_state_list.append((h, c))

        return self.final_layer(layer_output_list[-1][:,-1,:,:])



class ConvLSTMUpsample(nn.Module):
    def __init__(self, num_channels=3, feature_dim=64, lstm_hidden_dim=128):
        super(ConvLSTMUpsample, self).__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(num_channels, feature_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(input_size=feature_dim * 64 * 64, hidden_size=lstm_hidden_dim, batch_first=True)

        # Decoder (upsampling layers)
        self.decoder_conv1 = nn.ConvTranspose2d(lstm_hidden_dim, feature_dim, kernel_size=4, stride=2, padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(feature_dim, num_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()

        # Process each image in the sequence
        x = x.view(batch_size * seq_len, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Prepare for LSTM
        x = x.view(batch_size, seq_len, -1)

        # LSTM layer
        lstm_out, _ = self.lstm(x)
        print(lstm_out.shape)
        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]

        # Reshape and pass through the decoder
        lstm_out = lstm_out.view(batch_size, -1, 1, 1) 
        x = F.relu(self.decoder_conv1(lstm_out))
        x = self.decoder_conv2(x)

        return x
    
