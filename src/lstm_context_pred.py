import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMContextPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, device='cpu'):
        super(LSTMContextPredictor, self).__init__()
        self.hidden_size = hidden_size
        # self.lstm_hidden_size = lstm_hidden_size
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True).to(device)
        # self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dense = nn.Linear(hidden_size, out_size).to(device)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            # h0, c0 = torch.zeros(1, x.shape[0], self.lstm_hidden_size), torch.zeros(1, x.shape[0], self.lstm_hidden_size)
            h0, c0 = torch.zeros(1, x.shape[0], self.hidden_size), torch.zeros(1, x.shape[0], self.hidden_size)
            h0, c0 = h0.to(self.device), c0.to(self.device)
        if x.get_device() != h0.get_device():
            x = x.to(self.device)
        # print(f'x.shape={x.shape},{x.size()}, h0.shape={h0.shape}, {h0.size()}, c0.shape={c0.shape}, {c0.size()}')
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        output = self.dense(lstm_out)
        return output, (hn, cn)

# The GRUContextPredictor class can be merged with LSTMContextPredictor.
class GRUContextPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, device='cpu'):
        super(GRUContextPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = nn.GRU(input_size, hidden_size, 1, batch_first=True).to(device)
        self.dense = nn.Linear(hidden_size, out_size).to(device)

    def forward(self, x, h0=None):
        if h0 is None:
            h0 = torch.zeros(1, x.shape[0], self.hidden_size)
            h0 = h0.to(self.device)
        if x.get_device() != h0.get_device():
            x = x.to(self.device)
        lstm_out, hn = self.lstm(x, h0)
        out = self.dense(lstm_out)
        return out, hn

class RecurrentContextPredictor(nn.Module):
    def __init__(self, input_size, recurrent_hidden_size, hidden_layers, output_size, model_arch='lstm', dropout_prob=0.0, device='cpu'):
        super(RecurrentContextPredictor, self).__init__()
        # self.hidden_size = hidden_size
        self.recurrent_hidden_size = recurrent_hidden_size
        self.model_arch = model_arch
        self.device = device
        if model_arch == 'lstm':
            first_layer = nn.LSTM
        elif model_arch == 'gru':
            first_layer = nn.GRU
        else:
            raise ValueError(f'Invalid model_arch={model_arch}.')
        self.layers = nn.ModuleList([first_layer(input_size, recurrent_hidden_size, batch_first=True).to(device)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(p=dropout_prob)])
        if len(hidden_layers) > 0:
            self.layers.append(nn.Linear(recurrent_hidden_size, hidden_layers[0]).to(device))
            self.dropout_layers.append(nn.Dropout(p=dropout_prob))
            if len(hidden_layers) > 1:
                self.layers.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]).to(device) for i in range(1,len(hidden_layers))])
                self.dropout_layers.extend([nn.Dropout(p=dropout_prob) for i in range(1,len(hidden_layers))])
            self.layers.append(nn.Linear(hidden_layers[-1], output_size).to(device))
        else:
            self.layers.append(nn.Linear(recurrent_hidden_size, output_size).to(device))

    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(1, x.shape[0], self.recurrent_hidden_size)
            h0 = h0.to(self.device)
        if c0 is None and self.model_arch=='lstm':
            c0 = torch.zeros(1, x.shape[0], self.recurrent_hidden_size)
            c0 = c0.to(self.device)
        if x.get_device() != h0.get_device():
            x = x.to(self.device)
        if self.model_arch == 'lstm':
            x, (hn, cn) = self.layers[0](x, (h0, c0))
        elif self.model_arch == 'gru':
            x, hn = self.layers[0](x, h0)
        x = self.dropout_layers[0](x)
        for i in range(1, len(self.layers)-1):
            layer = self.layers[i]
            dropout_layer = self.dropout_layers[i]
            x = F.relu(layer(x))
            x = dropout_layer(x) 

        output_layer = self.layers[-1]
        output = output_layer(x)
        if self.model_arch == 'lstm':
            return output, (hn, cn)
        elif self.model_arch == 'gru':
            return output, hn

class MLPContextPredictor(nn.Module):
    def __init__(self, input_size, output_size, layers, device='cpu'):
        super(MLPContextPredictor, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, layers[0]).to(device)])
        self.linears.extend([nn.Linear(layers[i-1], layers[i]).to(device) for i in range(1,len(layers))])
        self.linears.append(nn.Linear(layers[-1], output_size).to(device))

    def forward(self, x):
        for i in range(len(self.linears)-1):
            layer = self.linears[i]
            x = F.relu(layer(x))
        output_layer = self.linears[-1]
        output = output_layer(x)
        return output