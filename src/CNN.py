from torch import nn

# Model define
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            
# Define 3D_CNN model class

class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.MaxPool3d(2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.MaxPool3d(2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96)
        )
        
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(768, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.flatten(x)
        return x
            
    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x