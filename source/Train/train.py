import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from source.Data.make_dataset import CustomDataset,data_loader
from source.Model import multimodal_attention
from source.Train import train_test_blocks
import utils
from logger import logging


# Create Dataset and Dataloaders
t0=transforms.Compose([transforms.Resize((224,224)),
                       transforms.ToTensor()])
transform_tr = [
    t0,
    transforms.Compose([transforms.RandomPerspective(distortion_scale=0.4, p=1.0), t0]),
    transforms.Compose([transforms.RandomRotation((0, 90)), t0]),
    transforms.Compose([transforms.RandomRotation((0, 180)), t0]),
    transforms.Compose([transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)), t0]),
    transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), t0]),
    transforms.Compose([transforms.RandomGrayscale(p=0.1), t0]),
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), t0]),
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0), t0]),
    transforms.Compose([transforms.RandomRotation(30, expand=False, center=None), t0]),
    transforms.Compose([transforms.RandomRotation(60, expand=False, center=None), t0]),
    transforms.Compose([transforms.RandomRotation(120, expand=False, center=None), t0]),
    transforms.Compose([transforms.RandomRotation(150, expand=False, center=None), t0]),
    transforms.Compose([transforms.RandomRotation(210,  expand=False, center=None), t0]),
    transforms.Compose([transforms.RandomRotation(240,  expand=False, center=None), t0]),
    transforms.Compose([transforms.RandomRotation(300,  expand=False, center=None), t0]),
    transforms.Compose([transforms.RandomRotation(330,  expand=False, center=None), t0]),
    transforms.Compose([transforms.RandomAffine(degrees=30, shear=30, translate=(0.1, 0.1), scale=(0.8, 1.2)), t0]),
    transforms.Compose([transforms.RandomAffine(degrees=60, shear=60, translate=(0.1, 0.1), scale=(0.8, 1.2)), t0]),
    transforms.Compose([transforms.RandomAffine(degrees=120, shear=120, translate=(0.1, 0.1), scale=(0.8, 1.2)), t0]),
    transforms.Compose([transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), t0]),
    transforms.Compose([transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5), t0]),
    transforms.Compose([transforms.RandomApply([transforms.GaussianBlur(kernel_size=7)], p=0.5), t0]),
    transforms.Compose([transforms.RandomApply([transforms.GaussianBlur(kernel_size=11)], p=0.5), t0]),
    transforms.Compose([transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.5), t0])
]
batch_size = 8
dataset = CustomDataset("source\Data\dataset_metadata_latest.csv", transform = transform_tr )
load_data = data_loader(dataset,batch_size=batch_size)
trainloader,testloader,valloader = load_data.create_dataloader()


# Define Network and network parameters
net = multimodal_attention.Multi_Modal_attention()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr = 0.00001)
net.to(device)
epoch = 20

traintest = train_test_blocks.train_test(trainloader=trainloader,testloader=testloader,valloader=valloader,net=net,device=device,optimizer=optimizer,criterion=criterion,batch_size = batch_size)

epoch_loss = []*epoch 
epoch_Accuracy = []*epoch
epoch_val_loss = []*epoch 
epoch_val_Accuracy = []*epoch


for epoch_index in range(epoch):
    print(f'Epoch: {epoch_index}\n')

    train_loss, train_accuracy = traintest.train_one_epoch()
    val_loss, val_accuracy = traintest.validate_one_epoch()

    # Append the values to the lists
    epoch_loss.append(train_loss)
    epoch_Accuracy.append(train_accuracy)

    # Optionally, you can also append validation values
    epoch_val_loss.append(val_loss)
    epoch_val_Accuracy.append(val_accuracy)

print('Finished Training')

# graphics = utils.graphics(epoch_loss,epoch_val_loss,epoch_Accuracy,epoch_val_Accuracy)
# graphics.generate_accuracy_curve()
# graphics.generate_loss_curve()

logging.info(f"test accuracy = {traintest.test_one_epoch()}")
utils.save_model(net,"model_1")