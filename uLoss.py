import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import data_Preprocess
from natsort import natsorted

def Fill_withMeasurements(out,MP,param):
    paths = data_Preprocess.getDataPath()
    MPsNames = data_Preprocess.findTargetFiles(paths,'Train',dataType='measurements')
    for file in natsorted(MPsNames):
        if str(param['measurements_num'])+'_points_measurements'+param['layout'] in file:
            MPName = file
            break
    measurements = data_Preprocess.loadMeasurements(MPName)
    for i in measurements:
        x = round(i[0].item()/param['mesh_size'][0])
        y = round(i[1].item()/param['mesh_size'][1])
        out[:,0,x,y] = MP[:,x,y]
        #print (MP[:,x,y])
        #print (MP[:,x,y].size())
    return out

def Fill_withDirechletBC(out,param):
    out[:,0,-1,0:51] = param['Direchlet BC']*torch.ones_like(out[:,0,-1,0:51])
    return out

def loss_neumann(out,param):
    hx = param['mesh_size'][0]
    hy = param['mesh_size'][1]
    coe_x = 2*hx*param['q']/param['Heat Transfer Rate']
    coe_y = 2*hy*param['q']/param['Heat Transfer Rate']
    MR = hy/hx
    f_ij = []
    if param['BCs'][0] == 'Neumann': # top
        T_ij = (2+2*MR**2)*out[:,0,1:-1,-1] - 2*out[:,0,1:-1,-2] - MR**2*(out[:,0,2:,-1]+out[:,0,0:-2,-1]) - coe_y*torch.ones_like(out[:,0,1:-1,-1])
        f_ij.append(T_ij) 
        #print ('top')
    if param['BCs'][1] == 'Neumann': # bottom
        T_ij = (2+2*MR**2)*out[:,0,1:-1,0] - 2*out[:,0,1:-1,1] - MR**2*(out[:,0,2:,0]+out[:,0,0:-2,0]) - coe_y*torch.ones_like(out[:,0,1:-1,0])
        f_ij.append(T_ij)
        #print ('bottom')
    if param['BCs'][2] == 'Neumann': # left
        T_ij = (2/MR**2+2)*out[:,0,0,1:-1] - 2*out[:,0,1,1:-1] - 1/MR**2*(out[:,0,0,2:]+out[:,0,0,0:-2]) - coe_x*torch.ones_like(out[:,0,0,1:-1])
        f_ij.append(T_ij)
        #print ('left')
    if param['BCs'][3] == 'Neumann': # right
        T_ij = (2/MR**2+2)*out[:,0,-1,1:-1] - 2*out[:,0,-2,1:-1] - 1/MR**2*(out[:,0,-1,2:]+out[:,0,-1,0:-2]) - coe_x*torch.ones_like(out[:,0,-1,1:-1])
        f_ij.append(T_ij)
        #print ('right')
    f_ij = torch.cat(f_ij)
    return torch.mean(torch.abs(f_ij))

def loss_radiation(out,param):
    sigma = 5.67e-8
    abs_zero = -273.15
    hx = param['mesh_size'][0]
    hy = param['mesh_size'][1]
    MR = hy/hx
    coe_x = 2*hx*param['Emissivity']*sigma/param['Heat Transfer Rate']
    coe_y = 2*hy*param['Emissivity']*sigma/param['Heat Transfer Rate']
    f_ij = []
    if param['BCs'][0] == 'Radiation': # top
        T_ij = 2*out[:,0,1:-1,-2] - (2+2*MR**2)*out[:,0,1:-1,-1] + MR**2*(out[:,0,2:,-1]+out[:,0,:-2,-1]) - coe_y*((out[:,0,1:-1,-1]-abs_zero)**4 - (param['T_a']-abs_zero)**4)
        f_ij.append(T_ij) 
        #print ('top')
    if param['BCs'][1] == 'Radiation': # bottom
        T_ij = 2*out[:,0,1:-1,1] - (2+2*MR**2)*out[:,0,1:-1,0] + MR**2*(out[:,0,2:,0]+out[:,0,:-2,0]) - coe_y*((out[:,0,1:-1,0]-abs_zero)**4 - (param['T_a']-abs_zero)**4)
        f_ij.append(T_ij)
        #print ('bottom')
    if param['BCs'][2] == 'Radiation': # left
        T_ij = 2*out[:,0,1,1:-1] - (2+2/MR**2)*out[:,0,0,1:-1] + 1/MR**2*(out[:,0,0,2:]+out[:,0,0,:-2]) - coe_x*((out[:,0,0,1:-1]-abs_zero)**4-(param['T_a']-abs_zero)**4)
        f_ij.append(T_ij)
        #print ('left')
    if param['BCs'][3] == 'Radiation': # right
        T_ij = 2*out[:,0,-2,1:-1] - (2+2/MR**2)*out[:,0,-1,1:-1] + 1/MR**2*(out[:,0,-1,2:]+out[:,0,-1,:-2]) - coe_x*((out[:,0,-1,1:-1]-abs_zero)**4-(param['T_a']-abs_zero)**4)
        f_ij.append(T_ij)
        #print ('right')
    f_ij = torch.cat(f_ij)    
    return torch.mean(torch.abs(f_ij))

def loss_mixBC(out,param):
    hx = param['mesh_size'][0]
    hy = param['mesh_size'][1]
    MR = hy/hx
    T_ij = (2/MR**2+2)*out[:,0,-1,52:-1] - 2*out[:,0,-2,52:-1] - 1/MR**2*(out[:,0,-1,53:]+out[:,0,-1,51:-2])
    losses = torch.mean(torch.abs(T_ij))  
    if param['BC_regular'] == True:
        Reg = (out[:,0,-1,51:]-out[:,0,-2,51:])/hx
        losses = losses + 0.05*torch.mean(torch.exp(-75000*Reg**2))
    return  losses       

def loss_convection(out,param):
    hx = param['mesh_size'][0]
    hy = param['mesh_size'][1]
    coe_x = 2*hx*param['Convective coe']/param['Heat Transfer Rate']
    coe_y = 2*hy*param['Convective coe']/param['Heat Transfer Rate']
    MR = hy/hx
    f_ij = []
    Reg = []
    if param['BCs'][0] == 'Convection': # top
        T_ij = (2*MR**2+2.0+coe_y)*out[:,0,1:-1,-1] - MR**2*(out[:,0,2:,-1]+out[:,0,0:-2,-1]) - 2*out[:,0,1:-1,-2] - coe_y*param['T_a']*torch.ones_like(out[:,0,1:-1,-1])
        f_ij.append(T_ij) 
        Reg.append((out[:,0,:,-1]-out[:,0,:,-2])/hy)           
        #print ('top')
    if param['BCs'][1] == 'Convection': # bottom
        T_ij = (2*MR**2+2.0+coe_y)*out[:,0,1:-1,0] - MR**2*(out[:,0,2:,0]+out[:,0,:-2,0]) - 2*out[:,0,1:-1,1] - coe_y*param['T_a']*torch.ones_like(out[:,0,1:-1,0])
        f_ij.append(T_ij)
        Reg.append((out[:,0,:,0]-out[:,0,:,1])/hy)
        #print ('bottom')
    if param['BCs'][2] == 'Convection': # left
        T_ij = (2/MR**2+2.0+coe_x)*out[:,0,0,1:-1] - (1/MR**2)*(out[:,0,0,2:]+out[:,0,0,0:-2]) - 2*out[:,0,1,1:-1] - coe_x*param['T_a']*torch.ones_like(out[:,0,0,1:-1])
        f_ij.append(T_ij)
        Reg.append((out[:,0,0,:]-out[:,0,1,:])/hx)
        #print ('left')
    if param['BCs'][3] == 'Convection': # right
        T_ij = (2/MR**2+2.0+coe_x)*out[:,0,-1,1:-1] - (1/MR**2)*(out[:,0,-1,2:]+out[:,0,-1,0:-2]) - 2*out[:,0,-2,1:-1] - coe_x*param['T_a']*torch.ones_like(out[:,0,-1,1:-1])
        f_ij.append(T_ij)
        Reg.append((out[:,0,-1,:]-out[:,0,-2,:])/hx)    
    f_ij = torch.cat(f_ij)
    Reg = torch.cat(Reg)
    losses = torch.mean(torch.abs(f_ij))
    if param['BC_regular'] == True:
        losses = losses + 0.01*torch.mean(torch.exp(-200 * Reg**2))
    return losses   

def loss_T_inner(out,param):
    hx = param['mesh_size'][0]
    hy = param['mesh_size'][1]
    T_ij = out[:,0,1:-1,1:-1]
    T_iplus1_j = out[:,0,2:,1:-1]
    T_iminus1_j = out[:,0,0:-2,1:-1]
    T_i_jplus1 = out[:,0,1:-1,2:]
    T_i_jminus1 = out[:,0,1:-1,0:-2]
    f_ij = hy**2*(T_iplus1_j+T_iminus1_j) + hx**2*(T_i_jplus1+T_i_jminus1) - 2*(hx**2+hy**2)*T_ij
    #print (torch.mean(torch.abs(f_ij)))
    return torch.mean(torch.abs(f_ij))

def loss_HSD(out,HSD,param):
    BC_D = param['HS_region']
    mesh_size = param['mesh_size']
    hx = mesh_size[0]
    hy = mesh_size[1]
    Q_err = []
    Regularity = []
    for i in BC_D:  
        preds = out[:,0,round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1,round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1].to(param['device'])
        trues = HSD[:,round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1,round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1].to(param['device'])
        Q_err.append(torch.flatten(trues-preds))
        if param['HS_regular'] is True:
            Regularity.append((torch.abs(preds-torch.mean(preds,dim=[1,2]).view(-1,1,1)*torch.ones_like(preds))).flatten())
    losses = torch.mean(torch.abs(torch.cat(Q_err)))
    if param['HS_regular'] is True:
        #print (torch.mean(torch.cat(Regularity)))
        losses = losses + torch.mean(torch.cat(Regularity))
    return losses

class UncertaintyLoss(nn.Module):
    def __init__(self,param):
        super().__init__()
        self.param = param
        self.task_num = len(self.param['BCs'])+2
        sigma = torch.randn(self.task_num)
        self.sigma = nn.Parameter(sigma)

    def forward(self,HSD,out,epoch):
        losses = 0
        self.loss_t_inner = loss_T_inner(out,self.param)
        self.loss_t_neumann = loss_neumann(out,self.param)
        self.loss_t_radiation = loss_radiation(out,self.param)
        self.loss_t_mixBC = loss_mixBC(out,self.param)
        self.loss_t_convection = loss_convection(out,self.param)
        self.loss_hsd = loss_HSD(out,HSD,self.param)
        if epoch < 500:
            losses = self.loss_hsd
        else:
            losses += (0.5/(self.sigma[0]**2)*self.loss_t_inner + torch.log(self.sigma[0]**2+1))           
            losses += (0.5/(self.sigma[1]**2)*self.loss_t_neumann + torch.log(self.sigma[1]**2+1))            
            losses += (0.5/(self.sigma[2]**2)*self.loss_t_radiation + torch.log(self.sigma[2]**2+1))
            losses += (0.5/(self.sigma[4]**2)*self.loss_t_convection + torch.log(self.sigma[4]**2+1))
            losses += (0.5/(self.sigma[5]**2)*self.loss_hsd + torch.log(self.sigma[5]**2+1))
        return losses
    def getLossComponent(self):
        #return self.w_t_inner*self.loss_t_inner,self.w_t_edge*self.loss_t_edge,self.w_hsd*self.loss_hsd
        return self.loss_t_inner, self.loss_t_neumann, self.loss_t_radiation, self.loss_t_convection, self.loss_hsd
    
