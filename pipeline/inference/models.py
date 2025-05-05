
import torch.nn as nn
import torch.nn.functional as F

#%%
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.03) # 0.03
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
            
class MLPResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(MLPResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        out += residual  # Skip connection
        out = self.activation(out)
        return out

class Generater_MLP_Skip(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_residual_blocks):
        super(Generater_MLP_Skip, self).__init__()
    
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.residual_blocks = nn.ModuleList([MLPResidualBlock(hidden_size) for _ in range(num_residual_blocks)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.activation = nn.LeakyReLU(0.2)
        self.activation = nn.Tanh()
        self.apply(weights_init_normal)
        
    def forward(self, x):
        out = self.activation(self.input_layer(x))
        
        for block in self.residual_blocks:
            out = block(out)
            
        out = self.output_layer(out)
        return out
    

