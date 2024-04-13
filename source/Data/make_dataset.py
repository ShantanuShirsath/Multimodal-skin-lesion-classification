import torch
import torch.utils
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import torch.utils.data
import sys
from exception import CustomException
from logger import logging



class CustomDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        try:
            self.data = pd.read_csv(csv_file)
            self.transform = transform
            self.load_data()
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod   
    def preprocess_radar(radar):
        try:
            real_p = torch.real(radar).float()
            imag_p = torch.imag(radar).float()
            # Stack real and imaginary parts along a new dimension
            radar_data = torch.stack((real_p, imag_p), dim=1)
            radar_data = radar_data.permute(1,0,2)
            return radar_data
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def to_mfcc(radar):
        try:
            magnitude_spectrum = np.absolute(radar)
            #log_magnitude = np.log(magnitude_spectrum)
            power_spectrum = np.fft.fft(magnitude_spectrum)
            return torch.tensor(power_spectrum).unsqueeze(0)
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def load_data(self):
        try:
            self.data_append = []
            for i in range(self.data.shape[0]):
                image = Image.open('B:\Arbeit\Data\skin_cropped_man\Images_skin\id_'+f'{i+1:02d}'+'.png').convert('RGB')
                label = self.data.iloc[i,-1]
                if type(self.transform) == list:
                    for j,tr in enumerate(self.transform):
                        transformed_image = tr(image)
                        radar_path = r'B:\Arbeit\Data\mmwave_csv\mmwave_new\Unzipped\CSV_files\id_'+f'{i+1:02d}'+'_sample_'+f'{i+1}'+'.csv'
                        radar_in = torch.tensor(np.genfromtxt(radar_path, delimiter=",", dtype=None, encoding=None))
                        radar_out = self.to_mfcc(radar_in.T)
                        label_ = label
                        self.data_append.append((transformed_image, radar_out, label_))
                else:
                    image = self.transform(image)
                    radar_ = pd.read_csv(r'B:\Arbeit\Data\mmwave_csv\mmwave_new\Unzipped\CSV_files\id_'+f'{i+1:02d}'+'_sample_'+f'{i+1}'+'.csv')
                    radar_out = self.to_mfcc(radar_).float()
                    label_ = label
                    self.data_append.append((image, radar_out , label_))
        except Exception as e:
            raise CustomException(e,sys)
        
    def __len__(self):
        return len(self.data_append)

    def __getitem__(self, idx):
        try:
            idx = idx % len(self.data_append)
            image, radar_data, label = self.data_append[idx]        
            return image, radar_data, label
        except Exception as e:
            raise CustomException(e,sys)
        

class data_loader():
    def __init__(self,dataset,batch_size):
        try:  
            self.dataset = dataset
            self.batch_size = batch_size
        except Exception as e:
            raise CustomException(e,sys)
    
    def create_dataloader(self):
        try:
            trainset,valset = torch.utils.data.random_split(self.dataset,[900,200])
            trainset,testset = torch.utils.data.random_split(trainset,[700,200])
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,shuffle=True)
            valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size,shuffle=True)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,shuffle=True)
            return trainloader,valloader,testloader
        except Exception as e:
            raise CustomException(e,sys)