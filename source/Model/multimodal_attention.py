from source.Model import preprocessor_blocks
import torch
import torch.nn as nn
from exception import CustomException
from logger import logging
import sys



class Multi_Modal_attention(nn.Module):
    def __init__(self):
        try:
            super(Multi_Modal_attention,self).__init__()
            self.embed_dim = 64 
            self.heads = 8
            self.radar_block = preprocessor_blocks.radar_block()
            self.Image_CNN = preprocessor_blocks.ImageCNNblock()
            self.attention_image = nn.MultiheadAttention(embed_dim= self.embed_dim, num_heads=self.heads)
            self.attention_sound = nn.MultiheadAttention(embed_dim=self.embed_dim,num_heads= self.heads)
            self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim,num_heads=self.heads)
            self.FC_final_block = preprocessor_blocks.FC_final_block()
        except Exception as e:
            raise CustomException(e,sys)
            
    
    def forward(self,image_input,radar_input):
        try:
            image_cnn_out = self.Image_CNN(image_input)
            radar_out = self.radar_block(radar_input.float())
            #print("Before sound:",radar_out.shape)
            image_cnn_out = torch.unsqueeze(image_cnn_out,dim=-1)
            image_cnn_out = image_cnn_out.permute(0,2,1)
            radar_out = torch.unsqueeze(radar_out,dim=-1)
            radar_out = radar_out.permute(0,2,1)
            #print("Final sound:",radar_out.shape)
            
            image_self_attention,_ = self.attention_image(image_cnn_out,image_cnn_out,image_cnn_out)
            sound_self_attention,_ = self.attention_sound(radar_out,radar_out,radar_out)
            
            #print(image_self_attention.shape)
            #print(sound_self_attention.shape)
            
            image_cross_attention,_ = self.cross_attention(image_self_attention,radar_out,radar_out)
            sound_cross_attention,_ = self.cross_attention(sound_self_attention,image_cnn_out,image_cnn_out)
            
            image_cross_attention = image_cross_attention.view(image_cross_attention.size(0), -1)
            sound_cross_attention = sound_cross_attention.view(sound_cross_attention.size(0), -1)
            #print(image_cross_attention.shape)
            #print(sound_cross_attention.shape)
            concat = torch.cat((image_cross_attention, sound_cross_attention), dim=1)
            #print(concat.shape)
            x = self.FC_final_block(concat)
            return x
        
        except Exception as e:
            raise CustomException(e,sys)