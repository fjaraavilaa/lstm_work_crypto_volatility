import torch
import torch.nn as nn

def generate_basic_lstm_input(batch_size=32, sequence_length=50, input_size=10):
    """
    Generate basic random input for LSTM.
    
    Args:
        batch_size (int): Number of sequences in batch
        sequence_length (int): Length of each sequence
        input_size (int): Dimensionality of input features
    Returns:
        torch.Tensor: Shape (batch_size, sequence_length, input_size)
    """
    # Simple random normal distribution
    x = torch.randn(batch_size, sequence_length, input_size)
    
    print(f"📊 Basic random input shape: {x.shape}")
    print(f"   Mean: {x.mean().item():.3f}")
    print(f"   Std: {x.std().item():.3f}")
    
    return x

stupid = generate_basic_lstm_input()


class LSTM_generalized_output(nn.Module):
    def __init__(self, step_size, input_dimensions, n_layers: int, hidden_sizes: list, output_dimensions: int, device: str):
      super(LSTM_generalized_output, self).__init__()
      self.hidden_sizes = hidden_sizes
      self.sizes = hidden_sizes.copy()
      self.sizes.insert(0, input_dimensions)
      self.step_size = step_size
      self.device = device
      self.input_dimensions = input_dimensions
      self.output_dimensions = output_dimensions
      self.hidden_layers = nn.ModuleList([nn.LSTM(input_size = self.sizes[i], 
                                                  hidden_size = self.sizes[i + 1], 
                                                  num_layers = 1, batch_first = True) for i in range(n_layers)])
      self.fc_layer = nn.Linear(self.step_size*self.hidden_sizes[hidden_sizes.__len__() - 1], self.output_dimensions)
    def forward(self, x):
      batch_size, seq_len, _ = x.size()
      i = 0
      for layer in self.hidden_layers:
        if i == 0:
           h_i = torch.zeros(1, batch_size, self.hidden_sizes[i]).to(self.device)
           c_i = torch.zeros(1, batch_size, self.hidden_sizes[i]).to(self.device)
           hidden_i = (h_i, c_i)
           x, hidden_i = layer(x, hidden_i)
        else:
            x, hidden_i = layer(x, hidden_i)
        i += 1
      x = x.contiguous().view(batch_size,-1)
      return self.fc_layer(x).unsqueeze(1)

lstm_test = LSTM_generalized_output(step_size=50, input_dimensions=10, 
                                    n_layers=2, hidden_sizes=[20, 30], 
                                    output_dimensions=5, device='cpu')