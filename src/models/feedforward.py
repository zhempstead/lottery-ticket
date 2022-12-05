import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFeedforward(nn.Module):
    """
    As defined in Frankle et al. 2019. In that paper it is called "LeNet." LeNet usually refers to
    a convolutional neural network defined in LeCun et al. 1998 but in Frankle it refers to a dense
    network also specified in that same paper.

    The layers are the same, but I use ReLu instead of tanh as the nonlinear terms as the former is
    generally recommended nowadays.
    """
    def __init__(self):
        super(SimpleFeedforward, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28 * 28, 300)  # 5*5 from image dimension
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        #torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = F.softmax(x, dim=1)
        return probs

    
    def correlate_nodes(self, f):
        '''
        Assign each hidden node a random multiplier based on the f factor:
        - 50% chance the multiplier is drawn from a uniform distribution between 1 and f
        - 50% chance the multiplier is drawn from a uniform distribution between 1/f and 1

        Multiply all weights going into/out of that node by the multiplier
        '''
        with torch.no_grad():
            params = dict(self.named_parameters())
            new_fc1, new_fc2 = correlate_layer(params['fc1.weight'], params['fc2.weight'], f)
            params['fc1.weight'].copy_(new_fc1)
            params['fc2.weight'].copy_(new_fc2)
            new_fc2, new_fc3 = correlate_layer(params['fc2.weight'], params['fc3.weight'], f)
            params['fc2.weight'].copy_(new_fc2)
            params['fc3.weight'].copy_(new_fc3)

def correlate_layer(input_layer, output_layer, f):
    node_count = input_layer.shape[0]
    node_weights = np.random.uniform(1.0, f, node_count)
    invert = np.random.choice(a=[True, False], size=node_count)
    node_weights = np.where(invert, 1 / node_weights, node_weights)
    new_input = np.multiply(node_weights, input_layer.T).T
    new_output = np.multiply(node_weights, output_layer)
    return new_input, new_output
