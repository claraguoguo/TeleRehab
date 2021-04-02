import torch.nn as nn

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''

  def __init__(self, input_size):
      super(MLP, self).__init__()
      self.input_size = input_size
      self.output_size = 1

      self.fc1 = nn.Linear(self.input_size, 64)
      self.fc2 = nn.Linear(64, 32)
      self.fc3 = nn.Linear(32, self.output_size)

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

      x = self.fc2(x)
      x = act_func(x)

      x = self.fc3(x)
      return x
