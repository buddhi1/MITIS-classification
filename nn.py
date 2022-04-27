
import torch.nn as nn
import torch.nn.functional as F

class IndoorResNetNetwork(nn.Module):

    def __init__(self):
        super(IndoorResNetNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.out = nn.Linear(in_features=512, out_features=67)

    def forward(self, x):
        x = self.dropout1(self.batchnorm1(F.relu(self.fc1(x))))
        x = self.batchnorm2(F.relu(self.fc2(x)))
        return F.log_softmax(self.out(x), dim=1)


class IndoorMnasnetNetwork(nn.Module):

    def __init__(self):
        super(IndoorMnasnetNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=1280, out_features=1024)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.out = nn.Linear(in_features=512, out_features=67)

    def forward(self, x):
        x = self.dropout1(self.batchnorm1(F.relu(self.fc1(x))))
        x = self.batchnorm2(F.relu(self.fc2(x)))
        return F.log_softmax(self.out(x), dim=1)


class IndoorResNetDeepNetwork(nn.Module):

    def __init__(self):
        super(IndoorResNetDeepNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(p=0.2)

        self.fc4 = nn.Linear(in_features=256, out_features=128)
        self.batchnorm4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(p=0.2)

        self.out = nn.Linear(in_features=128, out_features=67)

    def forward(self, x):
        x = self.dropout1(self.batchnorm1(F.relu(self.fc1(x))))
        x = self.dropout2(self.batchnorm2(F.relu(self.fc2(x))))
        x = self.dropout3(self.batchnorm3(F.relu(self.fc3(x))))
        x = self.dropout4(self.batchnorm4(F.relu(self.fc4(x))))

        return F.log_softmax(self.out(x), dim=1)



class IndoorMnasnetDeepNetwork(nn.Module):

    def __init__(self):
        super(IndoorMnasnetDeepNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=1280, out_features=1024)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(p=0.2)

        self.fc4 = nn.Linear(in_features=256, out_features=128)
        self.batchnorm4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(p=0.2)

        self.out = nn.Linear(in_features=128, out_features=67)


    def forward(self, x):
        x = self.dropout1(self.batchnorm1(F.relu(self.fc1(x))))
        x = self.dropout2(self.batchnorm2(F.relu(self.fc2(x))))
        x = self.dropout3(self.batchnorm3(F.relu(self.fc3(x))))
        x = self.dropout4(self.batchnorm4(F.relu(self.fc4(x))))

        return F.log_softmax(self.out(x), dim=1)