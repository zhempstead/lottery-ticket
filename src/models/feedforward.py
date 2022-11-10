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

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = F.softmax(x, dim=1)
        return probs
