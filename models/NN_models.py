import torch.nn as nn
import torch
class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''

  def __init__(self, input_size):
    super(MLP, self).__init__()
    self.input_size = input_size
    self.output_size = 1

    # self.fc1 = nn.Linear(self.input_size, 64)
    # self.fc2 = nn.Linear(64, 32)
    # self.fc3 = nn.Linear(32, self.output_size)

    hidden_size = (self.input_size + self.output_size)//2

    self.fc1 = nn.Linear(self.input_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, self.output_size)


  def forward(self, features):
      ###### 3 Hidden Layers

      # if self.act_func == 'tanh':
      #     func = nn.Tanh()
      #
      #
      # elif self.act_func == 'sigmoid':
      #     func = nn.Sigmoid()
      #
      # elif self.act_func == 'relu':
      act_func = nn.ReLU()

      x = self.fc1(features.float())
      x = act_func(x)

      # x = self.fc2(x)
      # x = act_func(x)

      x = self.fc3(x)
      return x

class LinearReg(nn.Module):
    def __init__(self, inputSize):
        super(LinearReg, self).__init__()
        self.linear = nn.Linear(inputSize, 1)

    def forward(self, x):
        out = self.linear(x)
        return out