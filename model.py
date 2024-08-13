import torch
import torch.nn as nn

class QuantumShellNet(nn.Module):
    def __init__(self):
        super(QuantumShellNet, self).__init__()

        # Define the layers here
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=2)

        self.fc1 = nn.Linear(8 * 8 * 8, 128)
        self.fc2 = nn.Linear(128 + 3, 64)
        self.fc3 = nn.Linear(64 + 3, 16)
        self.fc4 = nn.Linear(16 + 3, 1)

        self.dropout = nn.Dropout(0.3)  # Dropout probability

        self.activation = nn.ReLU()

    def forward(self, x, mass_num, atom_num, neutron_num):
        mass_num = mass_num.view(-1, 1)
        atom_num = atom_num.view(-1, 1)
        neutron_num = neutron_num.view(-1, 1)

        # Define the forward pass here
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        x = self.activation(self.conv2(x))
        x = self.dropout(x)
        x = self.activation(self.conv3(x))
        x = self.dropout(x)
        x = self.activation(self.conv4(x))
        x = self.dropout(x)
        x = self.activation(self.conv5(x))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = torch.cat((x, mass_num, atom_num, neutron_num), dim=1)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = torch.cat((x, mass_num, atom_num, neutron_num), dim=1)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = torch.cat((x, mass_num, atom_num, neutron_num), dim=1)
        x = self.fc4(x)
        return x

# Example usage:
# model = QuantumShellNet()
# print(model)
