import torch.nn as nn
import torch.nn.functional as F



class ImageCNNblock(nn.Module):
    def __init__(self):
        super(ImageCNNblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.fc1 = nn.Linear(in_features=512*14*14, out_features=256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = self.flatten(x)
        
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        
        x = self.fc3(x)
        
        return x  
        


class radar_block(nn.Module):
    def __init__(self):
        super(radar_block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels= 32, kernel_size= 3,padding=1, stride=2)  #(32*200*75)
        self.pool1 = nn.MaxPool2d(2,2) #(32*100*37)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size= 3,padding=1,stride=2) #(64*80*18)
        self.pool2 = nn.MaxPool2d(2,2) #(64*24*9)
        
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        
        self.fc1 = nn.Linear(in_features= 14400, out_features= 512)
        #self.drop1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(in_features=512, out_features= 256)
        #self.drop2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(in_features=256, out_features= 128)
        #self.drop3 = nn.Dropout(p=0.3)
        
        self.out = nn.Linear(in_features=128, out_features= 64)
        
        
    def forward(self,x):
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.flatten(x)
        #print(x.shape)
        
        x = F.relu(self.fc1(x))
        #x = self.drop1(x)
        
        x = F.relu(self.fc2(x))
        #x = self.drop2(x)
        
        x = F.relu(self.fc3(x))
        #x = self.drop3(x)
        
        x = self.out(x)
        return x
    
class FC_final_block(nn.Module):
    def __init__(self):
        super(FC_final_block,self).__init__()
        self.fc1 = nn.Linear(in_features= 128, out_features= 256)
        #self.drop1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(in_features=256, out_features= 128)
        #self.drop2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(in_features=128, out_features= 64)
        #self.drop3 = nn.Dropout(p=0.3)
        
        self.out = nn.Linear(in_features=64, out_features= 4)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        #x = self.drop1(x)
        
        x = F.relu(self.fc2(x))
        #x = self.drop2(x)
        
        x = F.relu(self.fc3(x))
        #x = self.drop3(x)
        
        x = self.out(x)
        
        return x