import torch
import torch.nn as nn
import torch.nn.functional as F
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns

class Sampling(nn.Module):
    def __init__(self, samp_layers,Nz):
        super(Sampling, self).__init__()
        self.samp_layers = nn.ModuleList(samp_layers)
        self.NZ = Nz

    def forward(self, x):
        x1 = torch.zeros(x.shape[0],x.shape[1],self.NZ)
        if self.samp_layers is not None:
            i=0
            for samp_layers in self.samp_layers:
                x1[:,i,:] = samp_layers(x[:,i,:])
                i=i+1
        return x1

class SampLayer(nn.Module):
    def __init__(self, d_model, Nz):
        super(SampLayer, self).__init__()
        self.d_model = d_model
        self.NZ = Nz
        self.fc = nn.Linear(self.d_model, self.NZ)

    def forward(self, x):
        x = self.fc(x)
        return x

class Sampling(nn.Module):
    def __init__(self, samp_layers,Nz):
        super(Sampling, self).__init__()
        self.samp_layers = nn.ModuleList(samp_layers)
        self.NZ = Nz

    def forward(self, x):
        x1 = torch.zeros(x.shape[0],x.shape[1],self.NZ)
        if self.samp_layers is not None:
            i=0
            for samp_layers in self.samp_layers:
                x1[:,i,:] = samp_layers(x[:,i,:])
                i=i+1
        return x1

class RecLinear(nn.Module):
    def __init__(self, samp_layers,Nz):
        super(RecLinear, self).__init__()
        self.samp_layers = nn.ModuleList(samp_layers)
        self.NZ = Nz

    def forward(self, x):
        x1 = torch.zeros(x.shape[0],x.shape[1],self.NZ)
        if self.samp_layers is not None:
            i=0
            for samp_layers in self.samp_layers:
                x1[:,i,:] = samp_layers(x[:,i,:])
                i=i+1
        return x1

class LinearLayer(nn.Module):
    def __init__(self, d_model, Nz):
        super(LinearLayer, self).__init__()
        self.d_model = d_model
        self.NZ = Nz
        self.fc = nn.Linear(self.d_model, self.NZ)

    def forward(self, x):
        x = self.fc(x)
        return x