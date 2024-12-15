import torch

class LSTMModel(torch.nn.Module):
    # referenced to https://wandb.ai/sauravmaheshkar/LSTM-PyTorch/reports/Using-LSTM-in-PyTorch-A-Tutorial-With-Examples--VmlldzoxMDA2NTA5
    # Assumes we do not use stacked LSTM
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # LSTM
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):

        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)

        # One time step
        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h0, c0))
        out = self.fc(final_hidden_state[-1])
        out = self.softmax(out) #TODO: see if activation is necessary
        return out