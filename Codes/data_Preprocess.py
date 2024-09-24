import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import scienceplots
from natsort import natsorted

plt.style.use('science')
plt.rcParams['text.usetex'] = False
seed = 123 
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def getDataPath():
    return os.path.abspath(os.path.join(os.getcwd(),'..')) + '/Datas/'

def findTargetFiles(paths,setType,dataType,fileType='.txt'):# setType='Train' or 'Test';dataType= 'Tfield' or 'measurements'
    matching_files = []
    for root, dirs, files in os.walk(paths):
        for file in natsorted(files):
            if (file.endswith(fileType)) and (setType in file) and (dataType in file):
                matching_files.append(os.path.join(root, file))
    return matching_files

def findTargetFiles_measurements(paths,setType,dataType,param,fileType='.txt'):
    fileNames = findTargetFiles(paths,setType,dataType)
    measurement_fileNames = []
    for file in natsorted(fileNames):
        if str(param['measurements_num'])+'_points' in file:
            measurement_fileNames.append(file)
    return measurement_fileNames

def loadData_fromFiles(Filename):
    path = Filename
    return np.loadtxt(path)

def transferToTensor(data):
    return torch.from_numpy(data).to(torch.float32)

def addGaussianNoise(Datas,params):
    torch.manual_seed(123)
    Datas = Datas.clone().to(params['device'])
    noise = torch.randn_like(Datas[:,-1]).to(params['device'])
    Datas[:,-1] += params['Lambda']*noise
    return Datas

def map_size_compute(params):
    map_size = []
    for i in range(len(params['structure_size'])):
        map_size.append(round(params['structure_size'][i]/params['mesh_size'][i]+1))
    return map_size

def load_measurementsInfo():
    fileName = getDataPath() +  'measurements.npz'
    loaded_data = np.load(fileName)
    measurements = {}
    measurements['3_points'] = loaded_data['3_points_coords']
    measurements['4_points'] = loaded_data['4_points_coords']
    measurements['5_points'] = loaded_data['5_points_coords']
    measurements['6_points'] = loaded_data['6_points_coords']
    measurements['7_points'] = loaded_data['7_points_coords']
    measurements['8_points'] = loaded_data['8_points_coords']
    measurements['9_points'] = loaded_data['9_points_coords']
    return measurements

def load_measurementsInfo_empirical():
    measurements = {}
    measurements['9_points'] = np.array([[7.1,0],[1.0,0],[0,6.2],[1.6,6.2],[9.8,9.2],[4.2,3.0],[8.0,5.1],[5.2,6.9],[5.4,2.0]])
    return measurements

def load_measurementsInfo_regular():
    measurements = {}
    measurements['9_points'] = np.array([[1.6,1.6],[1.6,5.0],[1.6,8.3],[5.0,1.6],[5.0,5.0],[5.0,8.3],[8.3,1.6],[8.3,5.0],[8.3,8.3]])
    return measurements

def generate_Map(fileName,param):
    Data = addGaussianNoise(transferToTensor(loadData_fromFiles(fileName)),param)
    map_size = map_size_compute(param)
    maps = torch.zeros(map_size)
    for i in Data:
        coord = []
        for j in range(len(i[:-1])):
            coord.append(i[j]/param['mesh_size'][j])
        maps[round(coord[0].item())][round(coord[1].item())] = i[-1]
        if param['is_plotInputMap']==True:
            plt.imshow(maps.cpu())
            plt.show()
            print (round(coord[0].item()),round(coord[1].item()))
            print (maps[round(coord[0].item())][round(coord[1].item())])
    return maps.to(param['device'])

def loadMeasurements(Filename):
    return transferToTensor(loadData_fromFiles(Filename))

def plotTemperature(T):
    plt.figure(figsize=(6,6))
    plt.imshow(T.cpu().detach().numpy(),cmap='rainbow')
    plt.colorbar()
    plt.title('Temperature_pred')
    plt.show()

def plotThermalSources(Source):
    plt.figure(figsize=(6,6))
    plt.imshow(Source.cpu().detach().numpy(),cmap='rainbow')
    plt.colorbar()
    plt.title('Thermal Source Distribution')
    plt.show()

def get_Tfields(setType):
    paths = getDataPath()
    TFieldsNames = findTargetFiles(paths,setType,dataType='Tfield')
    Tfields = []
    for i in TFieldsNames: 
        Tfields.append(transferToTensor(loadData_fromFiles(i)))
    return torch.stack(Tfields)

def generate_HeatSourceDistribution(data,param):
    mesh_size = param['mesh_size']
    map_size = map_size_compute(param)
    HSD = torch.zeros(map_size)
    hx = mesh_size[0]
    hy = mesh_size[1]
    count = 0
    for i in param['HS_region']:
        if count < 3:
            HSD[round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1,round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1] = data[count]*torch.ones_like(HSD[round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1,round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1])
        else:
            HSD[round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1,round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1] = data[count-1]*torch.ones_like(HSD[round(i[0][0].item()/hx):round(i[0][1].item()/hx)+1,round(i[1][0].item()/hy):round(i[1][1].item()/hy)+1])
        count = count + 1
    return HSD.to(param['device'])

def compute_temperature_gradient(temperature_field):
    # 获取温度场的维度
    nrows, ncols = temperature_field.shape
    
    # 初始化梯度场，每个点都有两个分量：x方向和y方向
    gradient_field = np.zeros((nrows, ncols, 2))
    
    # 计算内部点的梯度
    for i in range(1, nrows-1):
        for j in range(1, ncols-1):
            # x方向的温度梯度（水平方向）
            gradient_field[i, j, 0] = (temperature_field[i, j+1] - temperature_field[i, j-1]) / 2
            # y方向的温度梯度（垂直方向）
            gradient_field[i, j, 1] = (temperature_field[i+1, j] - temperature_field[i-1, j]) / 2

    # 处理边界，这里仅做示例，实际应用中可能需要根据边界条件进行适当调整
    # 左边界
    gradient_field[:, 0, 0] = temperature_field[:, 1] - temperature_field[:, 0]
    # 右边界
    gradient_field[:, -1, 0] = temperature_field[:, -1] - temperature_field[:, -2]
    # 上边界
    gradient_field[0, :, 1] = temperature_field[1, :] - temperature_field[0, :]
    # 下边界
    gradient_field[-1, :, 1] = temperature_field[-1, :] - temperature_field[-2, :]

    return gradient_field