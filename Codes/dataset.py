import torch
import numpy as np
import data_Preprocess 

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
'''
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,setType,params):
        super(MyDataset,self).__init__()
        self.params = params
        paths = data_Preprocess.getDataPath()
        # Heat Source Distribution
        loadsNames = data_Preprocess.findTargetFiles(paths,setType,dataType='loads')
        #print (loadsNames)
        loads = data_Preprocess.loadData_fromFiles(loadsNames[0])
        self.HSDs = []
        for i in loads:
            self.HSDs.append(data_Preprocess.generate_HeatSourceDistribution(i,params))
        self.HSDs = torch.stack(self.HSDs)
        # Measurements
        MPsNames = data_Preprocess.findTargetFiles_measurements(paths,setType,dataType='measurements',param = params)
        #print (MPsNames)
        self.MPs = []
        for i in MPsNames:
            self.MPs.append(data_Preprocess.generate_Map(i,self.params))
        self.MPs = torch.stack(self.MPs)
       
    def __getitem__(self,idx):
        input_variable = self.MPs[idx]
        output_variable = self.HSDs[idx]
        return input_variable,output_variable
    
    def __len__(self):
        if self.HSDs.size(0) != self.MPs.size(0):
            #print (self.HSDs.size())
            #print (self.MPs.size())
            print ("***Data Error! The lengths of the input variables and output variables are not equal***")
        return self.HSDs.size(0)
'''
    
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,setType,DrivenType,params):
        super(MyDataset,self).__init__()
        self.params = params
        paths = data_Preprocess.getDataPath()        
        # Measurements
        MPType = 'measurements' + params['layout']
        MPsNames = data_Preprocess.findTargetFiles_measurements(paths,setType,dataType=MPType,param = params)
        self.MPs = []
        for i in MPsNames:
            self.MPs.append(data_Preprocess.generate_Map(i,self.params))
        self.MPs = torch.stack(self.MPs)
        if DrivenType == 'Co-driven': # or 'Co-driven'
            # Heat Source Distribution
            loadsNames = data_Preprocess.findTargetFiles(paths,setType,dataType='loads')
            loads = data_Preprocess.loadData_fromFiles(loadsNames[0])
            self.HSDs = []
            for i in loads:
                self.HSDs.append(data_Preprocess.generate_HeatSourceDistribution(i,params))
            self.outputs = torch.stack(self.HSDs) 
        else:       
            # T_fields
            TFieldNames = data_Preprocess.findTargetFiles(paths,setType,'Tfield')
            self.TFields = []
            for i in TFieldNames:
                self.TFields.append(data_Preprocess.transferToTensor(data_Preprocess.loadData_fromFiles(i)))
            self.outputs = torch.stack(self.TFields).to(params['device'])

    def __getitem__(self,idx):
        input_variable = self.MPs[idx]
        output_variable = self.outputs[idx]
        return input_variable,output_variable
    
    def __len__(self):
        if self.MPs.size(0) != self.outputs.size(0):
            #print (self.HSDs.size())
            #print (self.MPs.size())
            print ("***Data Error! The lengths of the input variables and output variables are not equal***")
        return self.outputs.size(0)