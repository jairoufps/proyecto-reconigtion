#!/usr/bin/env python3
import pdb 
#pdb.set_trace()
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#librerias para cargar datos de manera eficiente
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import cv2
from pathlib import Path
import matplotlib.pyplot as plt # for plotting
from PIL import Image
from readNeuronal import Net
import torch.multiprocessing as mp


tranformadaTraining = transforms.Compose([
    
    transforms.Resize([int(256),int(256)]),
    transforms.RandomCrop([240, 240]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


dataTraining = torchvision.datasets.ImageFolder('./products_assets',transform=tranformadaTraining)
dataLoaderTraining = DataLoader(dataTraining,batch_size=130,shuffle=False)

clasesEntramiento = dataTraining.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

redneuronal = Net()

#Si se encuentra en gpu ejecuto paralelismo
def initParallel():

    if torch.cuda.device_count() > 1:
        redneuronal = nn.DataParallel(redneuronal)

if torch.cuda.device_count() > 1:
    redneuronal = nn.DataParallel(redneuronal,device_ids=[0,1,2,3,4,5,6])


redneuronal.to(device)


print('se ejecutara en '+str(device))

def mostrarImagen(img):
   
    img = img / 2 + 0.5   
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#permite guardar las clases 
"""
def loadClassesOfFolder():
    p = Path('./products_assets') 
    for x in p.iterdir():
        if x.is_dir():
            ubicacionDirectorio = str(x)
            listaNombreDirectorio = ubicacionDirectorio.split('/')
            
            
            clasesEntramiento.append(listaNombreDirectorio[1])
"""



def reduceLearningRatio(optimizer):
    for param in optimizer.param_groups:
        param['lr'] /= 10




def entrenamiento():

    NUMBER_EPOCHS = 128
    LEARNING_RATIO = 0.1
    lossFunction = nn.CrossEntropyLoss()
    optimizador = optim.SGD(redneuronal.parameters(),lr=LEARNING_RATIO, momentum=0.9)
    cantidadLosscalculado = 0
    lossTotal = 0

    for epoch in range(NUMBER_EPOCHS):
        if epoch == 35:
            reduceLearningRatio(optimizador)
        if epoch == 65:
            reduceLearningRatio(optimizador)
        if epoch == 95:
            reduceLearningRatio(optimizador) 


        dataTrainingIter = iter(dataLoaderTraining)
        for data,labels in dataTrainingIter:
            try:

                redneuronal.zero_grad()
                data,labels = Variable(data.float().to(device)),Variable(labels.to(device))
                ouput = redneuronal(data)
                loss = lossFunction(ouput,labels)
                loss.backward()
                optimizador.step()
                lossTotal+=loss.item()
                cantidadLosscalculado+=1
                #print('Me estoy entrenando en '+str(device))
                #print('labels de entramiento  '+str(clasesEntramiento[labels[0].item()]))
            except Exception as e:
                print(e)
        
        if epoch%5==0:
            try:

                lossPromedio = (lossTotal/cantidadLosscalculado)
                print('funcion de perdida promedio '+str(lossPromedio))
                print('cantidad de ciclos '+str(epoch+1))
                lossTotal = 0
                cantidadLosscalculado = 0
            except Exception as e:
                print(e)

def testDataTraining():
    folderImagenes = Path('./imagentest')
    valorBaseImagen = 65536
    valorWitdh = 256
    valorHeight = 256
    for folder in folderImagenes.iterdir():
        imagen = cv2.imread(str(folder))
        if imagen.size<valorBaseImagen:
            imagen = cv2.resize(imagen,(valorWitdh,valorHeight),cv2.INTER_CUBIC)
        else:
            imagen = cv2.resize(imagen,(valorWitdh,valorHeight),cv2.INTER_AREA)
        imagenTensor = torch.from_numpy(imagen)
        imagenTensor = imagenTensor.transpose(0,2)
        imagenTensor = imagenTensor.transpose(1,2)
        imagenTensor = imagenTensor.unsqueeze(1)
        imagenTensor = imagenTensor.transpose(0,1)
       
        ouput = redneuronal(Variable(imagenTensor.float().to(device)))
        _,prediccion = torch.max(ouput.data,1)
        print('La prediccion es :')
        print(clasesEntramiento[prediccion[0].item()])
        
        print('la ubicacion de la imagen es ')
        print(str(folder))
     





    
def saveModel():
    try:

        torch.save(redneuronal.state_dict(),'./modelo/modelo1.pt')
    except:
        p =Path('./modelo')
        p.mkdir(parents=True, exist_ok=True)
        torch.save(redneuronal.state_dict(),'./modelo/modelo1.pt')


def loadModel():
    ubicacion = 'modelo1.pt'
    try:
        redneuronal.load_state_dict(torch.load('./modelo/'+ubicacion))
        redneuronal.eval()
    except:
        
        print('archivo no existe')


def trainingWithThreads():
    mp.set_start_method('spawn')
    number_process = 3

    redneuronal.share_memory()
    process = []

    for rank in range(number_process):
        p = mp.Process(target=entrenamiento)
        p.start()
        process.append(p)
    
    for p in process:
        p.join()


#entrenamiento()
#saveModel()
#loadModel()
#testDataTraining()


if __name__ == '__main__':
    entrenamiento()
    saveModel()
    testDataTraining()
    
    """
    mp.set_start_method('spawn')
    number_process = 4

    redneuronal.share_memory()
    process = []

    for rank in range(number_process):
        p = mp.Process(target=entrenamiento)
        p.start()
        process.append(p)
    
    for p in process:
        p.join()
    
    saveModel()
    testDataTraining()
    """
    #saveModel()