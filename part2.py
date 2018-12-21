import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import cv2
import os
import torchvision
import torchvision.transforms as transforms
import pickle


#Iterate through either test/training directory
in_dir = "hw6_data/train"
src = "hw6_data/train"


Y1 = np.array([1, 0, 0, 0, 0])
Y1 = np.broadcast_to(Y1, (800,5))

Y2 = np.array([0, 1, 0, 0, 0])
Y2 = np.broadcast_to(Y2, (800,5))

Y3 = np.array([0, 0, 1, 0, 0])
Y3 = np.broadcast_to(Y3, (800,5))

Y4 = np.array([0, 0, 0, 1, 0])
Y4 = np.broadcast_to(Y4, (800,5))

Y5 = np.array([0, 0, 0, 0, 1])
Y5 = np.broadcast_to(Y5, (800,5))

Y = np.concatenate((Y1, Y2), axis=0)
Y = np.concatenate((Y, Y3), axis=0)
Y = np.concatenate((Y, Y4), axis=0)
Y = np.concatenate((Y, Y5), axis=0)
print(Y[800])

X = []
for root, dirs, files in os.walk(in_dir):
    #Iterate through each background directory
    for folder in dirs:
        in_dir = src + "/" + folder
        #Iterate through each file
        for file in os.listdir(in_dir):
            img = cv2.imread(os.path.join(in_dir, file))
            resized = cv2.resize(img, (60, 40))
            
            flattened = np.ravel(resized)
            flattened = flattened/256
            X.append(flattened)
            print(flattened[0:10])
            

#PIK = "nn.dat"
#    
#with open(PIK, "wb") as f:
#    pickle.dump(X, f)
#with open(PIK, "rb") as f:
#    print(pickle.load(f))
    
#X = pickle.load(open("nn.dat", "rb"))

#Input layer is the size of the flattened input
N0 = 7200

#Hidden layers are arbitrarily chosen for now, should be around 2/3 of the input layer
N1 = 1000
N2 = 1000

#Output layer is the number of classifiers, in this case just 5
Nout = 5

class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
    		#Create three fully connected layers, two of which are hidden and the third is
    		#the output layer.  In each case, the first argument o Linear is the number
    		#of input values from the previous layer, and the second argument is the number
    		#of nodes in this layer.  The call to the Linear initializer creates a PyTorch
    		#functional that in turn adds a weight matrix and a bias vector to the list of
    		#(learnable) parameters stored with each Net object.  These weight matrices
        #bias vectors are implicitly initialized using a normal distribution
    		#with mean 0 and variance 1
        self.fc1 = nn.Linear(N0, N1, bias=True) 
        self.fc2 = nn.Linear(N1, N2, bias=True) 
        self.fc3 = nn.Linear(N2, Nout, bias=True)
        
        
    #The forward method takes an input Variable and creates a chain of Variables
    #from the layers of the network defined in the initializer. The F.relu is
	 #a functional implementing the Rectified Linear activation function.
    def forward(self, x):
		   x = F.relu(self.fc1(x)) 
		   x = F.relu(self.fc2(x)) 
		   x = self.fc3(x)
		   print(x.data)
		   return x
        
        
#  Create an instance of this network.
net = Net()


#  Set parameters to control the process
epochs = 1000
batch_size = 8
n_train = len(X)
n_batches = int(np.ceil(n_train / batch_size))
learning_rate = 1e-5
criterion = nn.MSELoss()

#test = net(X[0:1])
#print(test)

X, Y = Variable(torch.Tensor(X)), Variable(torch.Tensor(Y))

print(X)
#print(X[0])

print(X.shape, Y.shape )

for epoch in range(epochs):
    indices = torch.randperm(n_train)
    
    for b in range(n_batches):
        
        batch_indices = indices[b*batch_size: (b+1)*batch_size]
        batch_X = X[batch_indices]
        batch_Y = Y[batch_indices]
        
        pred_Y = net(batch_X)
        
        loss = criterion(pred_Y, batch_Y)
        
        net.zero_grad()
        loss.backward()
        
        for param in net.parameters():
            param.data -= learning_rate * param.grad.data
    

#pred_Y_train = net(X)
#print('Training success rate:', success_rate(pred_Y_train, Y_train))
#print('Test success rate:', success_rate(pred_Y_test, Y_test))
        
    
    
    
    
    
    
    
    
    
    
    
    