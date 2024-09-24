import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use('science')
plt.rcParams['text.usetex'] = False

import data_Preprocess
import uLoss

seed = 123 #seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class _EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, polling=True):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),#GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),#GELU(),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(8, middle_channels),
            nn.ReLU(),#GELU(),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),#GELU()
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, in_channels,out_channels, factors,param):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(in_channels, 8 * factors, polling=False)
        self.enc2 = _EncoderBlock(8 * factors, 16 * factors)
        self.enc3 = _EncoderBlock(16 * factors, 32 * factors)
        self.enc4 = _EncoderBlock(32 * factors, 64 * factors)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(64 * factors, 128 * factors, 64 * factors)
        self.dec4 = _DecoderBlock(128 * factors, 64 * factors, 32 * factors)
        self.dec3 = _DecoderBlock(64 * factors, 32 * factors, 16 * factors)
        self.dec2 = _DecoderBlock(32 * factors, 16 * factors, 8 * factors)
        self.dec1 = nn.Sequential(
            nn.Conv2d(16 * factors, 8 * factors, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(8, 8 * factors),
            nn.ReLU(),#GELU(),
            nn.Conv2d(8 * factors, 8 * factors, kernel_size=1, padding=0),
            nn.GroupNorm(8, 8 * factors),
            nn.ReLU(),#GELU(),
        )
        self.final = nn.Conv2d(8 * factors, out_channels, kernel_size=1)
        self.param = param
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)+self.param['T_a']
        return final

def generate_Loss_Comp_Initial_w():
    w_inner = 400
    w_neumann = 30
    w_radiation = 4
    w_mixBC = 20
    w_convection = 30
    w_hsd = 30
    return w_inner,w_neumann,w_radiation,w_mixBC,w_convection,w_hsd

def lr_weight_Cal(loss_t_inner, loss_t_neumann, loss_t_radiation, loss_t_convection, loss_hsd):
    weight = 1.0
    if loss_t_inner<5e-3:
        weight = weight * 0.5
    if loss_t_neumann<5e-3:
        weight = weight * 0.5
    if loss_t_radiation<5e-3:
        weight = weight * 0.5
    if loss_t_convection<5e-3:
        weight = weight * 0.5
    if loss_hsd<0.5:
        weight = weight * 0.5
    return weight

def AdaptiveWeight_comp(w,loss_t_1,loss_t):
    return w*(loss_t/loss_t_1)

def _Train(model,data_loader,loss_fn,param):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_losses_epoch = []
    loss_inner_epoch = []
    loss_neumann_epoch = []
    loss_radiation_epoch = []
    loss_convection_epoch = []
    loss_hsd_epoch = []
    since = time.time()
    model.train()
    lr_w = 1.0
    all_parameters = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = optim.Adam(all_parameters, lr_w*param['learning_rate'])
    for epoch in range(param['epochs']):
        train_losses_batch = 0
        loss_inner_batch = 0
        loss_neumann_batch = 0
        loss_radiation_batch = 0
        loss_convection_batch = 0
        loss_hsd_batch = 0
        #all_parameters = list(model.parameters()) + list(loss_fn.parameters())
        #optimizer = optim.Adam(all_parameters, lr_w*param['learning_rate'])
        for idx,data in enumerate(data_loader):
            MPs = data[0].clone().detach().requires_grad_(True)
            HSD = data[1].clone().detach().requires_grad_(True)
            optimizer.zero_grad()
            out = model(MPs.unsqueeze(1))
            out = uLoss.Fill_withMeasurements(out,MPs,param)
            #out = uLoss.Fill_withDirechletBC(out,param)

            loss = loss_fn(HSD,out,epoch)
            loss_t_inner,loss_t_neumann,loss_t_radiation,loss_t_convection,loss_hsd = loss_fn.getLossComponent()
            train_losses_batch += loss.item()
            loss_inner_batch += loss_t_inner.item()
            loss_neumann_batch += loss_t_neumann.item()
            loss_radiation_batch += loss_t_radiation.item()
            loss_convection_batch += loss_t_convection.item()
            loss_hsd_batch += loss_hsd.item()

            loss.backward()
            optimizer.step()
        
        train_losses_mean = train_losses_batch / len(data_loader)
        loss_inner_mean = loss_inner_batch / len(data_loader)
        loss_neumann_mean = loss_neumann_batch/len(data_loader)
        loss_radiation_mean = loss_radiation_batch / len(data_loader)
        loss_convection_mean = loss_convection_batch / len(data_loader)
        loss_hsd_mean = loss_hsd_batch / len(data_loader)

        #lr_w = lr_weight_Cal(loss_inner_mean,loss_neumann_mean,loss_radiation_mean,loss_convection_mean,loss_hsd_mean)
        
        train_losses_epoch.append(train_losses_mean)
        loss_inner_epoch.append(loss_inner_mean)
        loss_neumann_epoch.append(loss_neumann_mean)
        loss_radiation_epoch.append(loss_radiation_mean)
        loss_convection_epoch.append(loss_convection_mean)
        loss_hsd_epoch.append(loss_hsd_mean)

        if (epoch % param['dis_interval'] == 0) or (epoch + 1 == param['epochs']):
            print ('Traing Epoch: {}, learning_rate = {}'.format(epoch,lr_w*param['learning_rate']))
            print ('loss = {:.6f}, loss_hs = {:.6f}, loss_inner = {:.6f}'.format(train_losses_mean,loss_hsd_mean,loss_inner_mean))
            print ('loss component -> loss_neumann = {:.6f}, loss_radiation = {:.6f}, loss_convection = {:.6f}'.format(loss_neumann_mean,loss_radiation_mean,loss_convection_mean))
        
    print ('***********************  Train is done  ***********************')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    if param['is_plotLoss'] is True:
        plt.figure(figsize=(10,4))
        plt.plot(range(epoch+1),train_losses_epoch,'-*',color='black',linewidth=0.5,markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    return train_losses_epoch,loss_inner_epoch,loss_neumann_epoch,loss_radiation_epoch,loss_convection_epoch,loss_hsd_epoch

def _Train_DD(model,data_loader,loss_fn,param):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_losses_epoch = []
    since = time.time()
    model.train()
    lr_w = 1.0
    optimizer = optim.Adam(model.parameters(), lr_w*param['learning_rate'])
    for epoch in range(param['epochs']):
        train_losses_batch = 0
        for idx,data in enumerate(data_loader):
            MPs = data[0].clone().detach().requires_grad_(True)
            outputs = data[1].clone().detach().requires_grad_(True)
            optimizer.zero_grad()
            out = model(MPs.unsqueeze(1))
            loss = loss_fn(outputs.unsqueeze(1),out)
            train_losses_batch += loss.item()

            loss.backward()
            optimizer.step()
        
        train_losses_mean = train_losses_batch / len(data_loader)
        train_losses_epoch.append(train_losses_mean)

        if (epoch % param['dis_interval'] == 0) or (epoch + 1 == param['epochs']):
            print ('Traing Epoch: {}, loss = {}'.format(epoch,train_losses_mean))
        
    print ('***********************  Train is done  ***********************')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    if param['is_plotLoss'] is True:
        plt.figure(figsize=(10,4))
        plt.plot(range(epoch+1),train_losses_epoch,'-*',color='black',linewidth=0.5,markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    return train_losses_epoch
'''
class inversePINNs():
    def __init__(self,loads,setType,param):
        self.loss_function = nn.MSELoss(reduction ='mean')
        loads = loads.to(param['device'])  
        self.loads = nn.Parameter(loads)
        self.model = UNet(in_channels=1, out_channels=1, factors=2,param=param).to(param['device'])
        self.model.register_parameter('loads',self.loads)
        self.param = param
        self.setType = setType
        self.task_num = 4+2
        sigma = torch.randn(self.task_num).to(param['device'])
        self.sigma = nn.Parameter(sigma)

    def loss_data(self,SampleNum):
        paths = data_Preprocess.getDataPath()  
        MPsNames = data_Preprocess.findTargetFiles_measurements(paths,self.setType,dataType='measurements',param = self.param)
        Data = data_Preprocess.addGaussianNoise(data_Preprocess.transferToTensor(data_Preprocess.loadData_fromFiles(MPsNames[SampleNum])),self.param)
        Trues = Data[:,-1]
        self.MPs = data_Preprocess.generate_Map(MPsNames[SampleNum],self.param).unsqueeze(0)
        self.out = self.model(self.MPs.unsqueeze(0))
        Preds = []
        for i in Data:
            coord = []
            for j in range(len(i[:-1])):
                coord.append(i[j]/self.param['mesh_size'][j])
            Preds.append(self.out[0][0][round(coord[0].item())][round(coord[1].item())])
        Preds = torch.stack(Preds)
        loss_u = self.loss_function(Trues,Preds)
        return loss_u
    
    def loss_PDE(self):
        mesh_size = self.param['mesh_size']
        hx = mesh_size[0]
        hy = mesh_size[1] 
        adjusted_out = self.out.clone().detach().requires_grad_(True)    
        count = 0
        for i in self.param['HS_region']:
            if count < 3:
                adjusted_out[round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1,round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1] = self.loads[0][count].item()*torch.ones_like(self.out[round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1,round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1])
            else:
                adjusted_out[round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1,round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1] = self.loads[0][count-1].item()*torch.ones_like(self.out[round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1,round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1])
            count = count + 1
        losses = 0
        self.loss_t_inner = uLoss.loss_T_inner(adjusted_out,self.param)
        losses += (0.5/(self.sigma[0]**2)*self.loss_t_inner + torch.log(self.sigma[0]**2+1))  

        self.loss_t_neumann = uLoss.loss_neumann(adjusted_out,self.param)
        losses += (0.5/(self.sigma[1]**2)*self.loss_t_neumann + torch.log(self.sigma[1]**2+1))  
        
        self.loss_t_radiation = uLoss.loss_radiation(adjusted_out,self.param)
        losses += (0.5/(self.sigma[2]**2)*self.loss_t_radiation + torch.log(self.sigma[2]**2+1))  
        
        self.loss_t_convection = uLoss.loss_convection(adjusted_out,self.param) 
        losses += (0.5/(self.sigma[3]**2)*self.loss_t_convection + torch.log(self.sigma[3]**2+1))

        return losses
    
    def loss(self,SampleNum):
        losses = 0
        loss_d = self.loss_data(SampleNum)
        losses += (0.5/(self.sigma[4]**2)*loss_d + torch.log(self.sigma[4]**2+1))
        loss_p = self.loss_PDE()
        losses += (0.5/(self.sigma[5]**2)*loss_p + torch.log(self.sigma[5]**2+1))
        return losses


'''
class inversePINNs():
    def __init__(self,loads,setType,param):
        self.loss_function = nn.MSELoss(reduction ='mean')
        loads = loads.to(param['device'])
        self.loads = nn.Parameter(loads)
        self.model = UNet(in_channels=1, out_channels=1, factors=2,param=param).to(param['device'])
        self.model.register_parameter('loads',self.loads)
        self.param = param
        self.setType = setType
        self.task_num = 4+2
        sigma = torch.randn(self.task_num).to(param['device'])
        self.sigma = nn.Parameter(sigma)

    def loss_data(self,SampleNum):
        paths = data_Preprocess.getDataPath()  
        MPsNames = data_Preprocess.findTargetFiles_measurements(paths,self.setType,dataType='measurements',param = self.param)
        Data = data_Preprocess.addGaussianNoise(data_Preprocess.transferToTensor(data_Preprocess.loadData_fromFiles(MPsNames[SampleNum])),self.param)
        Trues = Data[:,-1]
        self.MPs = data_Preprocess.generate_Map(MPsNames[SampleNum],self.param).unsqueeze(0)
        self.out = self.model(self.MPs.unsqueeze(0))
        Preds = []
        for i in Data:
            coord = []
            for j in range(len(i[:-1])):
                coord.append(i[j]/self.param['mesh_size'][j])
            Preds.append(self.out[0][0][round(coord[0].item())][round(coord[1].item())])
        Preds = torch.stack(Preds)
        loss_u = self.loss_function(Trues,Preds)
        return loss_u
    
    def loss_PDE(self):
        mesh_size = self.param['mesh_size']
        hx = mesh_size[0]
        hy = mesh_size[1] 
        adjusted_out = self.out.clone()
        load = self.loads*80.0+20.0
        count = 0
        for i in self.param['HS_region']:
            if count < 3:
                value = load[0][count]
            else:
                value = load[0][count-1]
            mask = torch.zeros_like(adjusted_out)
            mask[:, :, round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1, round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1] = 1
            adjusted_out = adjusted_out * (1 - mask) + value * mask
            count = count + 1
        losses = 0
        self.loss_t_inner = uLoss.loss_T_inner(adjusted_out,self.param)
        losses += (0.5/(self.sigma[0]**2)*self.loss_t_inner + torch.log(self.sigma[0]**2+1))  

        self.loss_t_neumann = uLoss.loss_neumann(adjusted_out,self.param)
        losses += (0.5/(self.sigma[1]**2)*self.loss_t_neumann + torch.log(self.sigma[1]**2+1))  
        
        self.loss_t_radiation = uLoss.loss_radiation(adjusted_out,self.param)
        losses += (0.5/(self.sigma[2]**2)*self.loss_t_radiation + torch.log(self.sigma[2]**2+1))  
        
        self.loss_t_convection = uLoss.loss_convection(adjusted_out,self.param) 
        losses += (0.5/(self.sigma[3]**2)*self.loss_t_convection + torch.log(self.sigma[3]**2+1))

        return losses
    
    def loss(self,SampleNum):
        losses = 0
        loss_d = self.loss_data(SampleNum)
        losses += (0.5/(self.sigma[4]**2)*loss_d + torch.log(self.sigma[4]**2+1))
        loss_p = self.loss_PDE()
        losses += (0.5/(self.sigma[5]**2)*loss_p + torch.log(self.sigma[5]**2+1))
        return losses


    
def _Predict(model,maps,param):
    out = model(maps.unsqueeze(0))
    #T = uLoss.BC_Dirichlet(param,T,Source)
    T = uLoss.Fill_withMeasurements(T,param)
    if param['is_plotResult'] is True:
        data_Preprocess.plotTemperature(T)
    return T


'''
class inversePINNs():
    def __init__(self,loads,setType,param):
        self.loss_function = nn.MSELoss(reduction ='mean')
        loads = loads.to(param['device'])
        self.loads = nn.Parameter(loads)
        self.model = models.UNet(in_channels=1, out_channels=1, factors=2,param=param).to(param['device'])
        self.model.register_parameter('loads',self.loads)
        self.param = param
        self.setType = setType
        self.task_num = 4+2
        sigma = torch.randn(self.task_num).to(param['device'])
        self.sigma = nn.Parameter(sigma)

    def loss_data(self,SampleNum):
        paths = data_Preprocess.getDataPath()  
        MPsNames = data_Preprocess.findTargetFiles_measurements(paths,self.setType,dataType='measurements',param = self.param)
        Data = data_Preprocess.addGaussianNoise(data_Preprocess.transferToTensor(data_Preprocess.loadData_fromFiles(MPsNames[SampleNum])),self.param)
        Trues = Data[:,-1]
        self.MPs = data_Preprocess.generate_Map(MPsNames[SampleNum],self.param).unsqueeze(0)
        self.out = self.model(self.MPs.unsqueeze(0))
        Preds = []
        for i in Data:
            coord = []
            for j in range(len(i[:-1])):
                coord.append(i[j]/self.param['mesh_size'][j])
            Preds.append(self.out[0][0][round(coord[0].item())][round(coord[1].item())])
        Preds = torch.stack(Preds)
        loss_u = self.loss_function(Trues,Preds)
        return loss_u
    
    def loss_PDE(self):
        mesh_size = self.param['mesh_size']
        hx = mesh_size[0]
        hy = mesh_size[1] 
        adjusted_out = self.out.clone()
        load = self.loads*80.0+20.0
        count = 0
        for i in self.param['HS_region']:
            if count < 3:
                value = load[0][count]
            else:
                value = load[0][count-1]
            mask = torch.zeros_like(adjusted_out)
            mask[:, :, round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1, round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1] = 1
            adjusted_out = adjusted_out * (1 - mask) + value * mask
            count = count + 1
        plt.imshow(adjusted_out.cpu().view(101,101).detach().numpy(),cmap='rainbow')
        plt.colorbar()
        plt.title('after loss_p')
        plt.show()
        losses = 0
        self.loss_t_inner = uLoss.loss_T_inner(adjusted_out,self.param)
        losses += (0.5/(self.sigma[0]**2)*self.loss_t_inner + torch.log(self.sigma[0]**2+1))  

        self.loss_t_neumann = uLoss.loss_neumann(adjusted_out,self.param)
        losses += (0.5/(self.sigma[1]**2)*self.loss_t_neumann + torch.log(self.sigma[1]**2+1))  
        
        self.loss_t_radiation = uLoss.loss_radiation(adjusted_out,self.param)
        losses += (0.5/(self.sigma[2]**2)*self.loss_t_radiation + torch.log(self.sigma[2]**2+1))  
        
        self.loss_t_convection = uLoss.loss_convection(adjusted_out,self.param) 
        losses += (0.5/(self.sigma[3]**2)*self.loss_t_convection + torch.log(self.sigma[3]**2+1))
        
        return losses
    
    def loss(self,SampleNum):
        losses = 0
        loss_d = self.loss_data(SampleNum)
        plt.imshow(self.out.cpu().view(101,101).detach().numpy(),cmap='rainbow')
        plt.colorbar()
        plt.title('after loss_u')
        plt.show()
        losses += (0.5/(self.sigma[4]**2)*loss_d + torch.log(self.sigma[4]**2+1))
        loss_p = self.loss_PDE()
        losses += (0.5/(self.sigma[5]**2)*loss_p + torch.log(self.sigma[5]**2+1))
        return losses

'''