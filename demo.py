import pandas as pd
from PIL import Image,ImageFile
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import tqdm
import os
import time
device=torch.device('cuda:0')
ImageFile.LOAD_TRUNCATED_IMAGES=True
class MyDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__(csv_file)
        self.data=pd.read_csv(csv_file)
    def __getitem__(self, index):
        image=Image.open(os.path.join('../input/',self.data.get[index,'id_code']+'.png'))
        image.resize((256,256),resample=Image.BILINEAR)
        return {'image':image,'label':torch.Tensor(self.data.get[index,'label'])}
    def __len__(self):
        return len(self.data)
model=torchvision.models.resnet101(pretrained=False)
model.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))
num_features=model.fc.in_features
model.fc=nn.Linear(2048,1)
model=model.to(device)
train_dataset=MyDataset('./train.csv')
data_loader=DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=4)
plist = [
         {'params': model.layer4.parameters(), 'lr': 1e-4, 'weight': 0.001},
         {'params': model.fc.parameters(), 'lr': 1e-3}
         ]
optimizer=optim.Adam(plist,lr=0.001)
scheduler=lr_scheduler.StepLR(optimizer=optimizer,step_size=10)
since=time.time()
creterion=nn.MSELoss()
num_epoch=15
for epoch in range(num_epoch):
    print('epoch:{}/{}'.format(epoch,num_epoch-1))
    print('-'*10)
    scheduler.step()
    training_loss=0.0
    model.train()
    tk0=tqdm(data_loader,total=len(data_loader))
    for i,d in enumerate(tk0):
        inputs=d['image']
        labels=d['label']
        inputs=inputs.to(device,dtype=torch.float)
        labels=labels.to(device,dtype=torch.float)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs=model(inputs)
            loss=creterion(outputs,labels)
            
