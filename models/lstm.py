import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, max_frames):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(max_frames)

    def forward(self, inputs, inputs_lens):
        inputs = inputs.float()
        ## Note: inputs.data.shape = [ batch_size x max_seq_len x num_joints ]

        ## Example for pack_padded_sequence & pad_packed_sequence:
        ## https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec

        ## Step 1: Call pack_padded_sequence with padded] instances and sequence lengths
        inputs_packed = pack_padded_sequence(inputs, inputs_lens.cpu().numpy(), batch_first=True, enforce_sorted=False)
        ## Note: packed_inputs.data.shape = [ sum_of_lengths X num_joints ], where sum_of_lengths = SUM(inputs_lens)

        ## Step 2: Forward with LSTM
        output_packed, (hn, cn) = self.lstm(inputs_packed)

        ## Step 3: Call unpack_padded_sequences
        output_unpacked, lens_unpacked = pad_packed_sequence(output_packed, batch_first=True)

        ## Reference: https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
        out_forward = output_unpacked[range(len(output_unpacked)), inputs_lens - 1, :self.hidden_dim]
        out_reverse = output_unpacked[:, 0, self.hidden_dim:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        output = self.fc(out_reduced)
        ## Return the probability of being Class 1
        return output