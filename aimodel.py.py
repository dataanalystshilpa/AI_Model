#!/usr/bin/env python
# coding: utf-8

# In[54]:


import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time


# In[55]:


x, y = torch.load(r'C:\Users\Digisnare\Desktop\MNIST\processed\training.pt')


# In[56]:


y.shape


# In[57]:


plt.imshow(x[2].numpy())
plt.title(f'Number is {y[2].numpy()}')
plt.colorbar()
plt.show()


# In[58]:


y_original = torch.tensor([2, 4, 3, 0, 1])
y_new = F.one_hot(y_original)


# In[59]:


y_original


# In[60]:


y_new


# In[61]:


y


# In[62]:


y_new


# In[63]:


y_new = F.one_hot(y, num_classes=10)
y_new.shape


# In[64]:


x.shape


# In[65]:


x.view(-1,28**2).shape


# # PyTorch Dataset Object

# In[66]:


class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 255.
        self.y = F.one_hot(self.y, num_classes=10).to(float)
    def __len__(self): 
        return self.x.shape[0]
    def __getitem__(self, ix): 
        return self.x[ix], self.y[ix]


# In[67]:


train_ds = CTDataset(r'C:\Users\Digisnare\Desktop\MNIST\processed\training.pt')
test_ds = CTDataset(r'C:\Users\Digisnare\Desktop\MNIST\processed\training.pt')


# In[68]:


len(train_ds)


# In[69]:


xs, ys = train_ds[0:4]


# In[70]:


ys.shape


# # PyTorch DataLoader Object

# In[71]:


train_dl = DataLoader(train_ds, batch_size=5)


# In[72]:


for x, y in train_dl:
    print(x.shape)
    print(y.shape)
    break


# In[73]:


len(train_dl)


# # Cross Entropy Loss

# In[74]:


L = nn.CrossEntropyLoss()


# # The Network

# In[75]:


class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2,100)
        self.Matrix2 = nn.Linear(100,50)
        self.Matrix3 = nn.Linear(50,10)
        self.R = nn.ReLU()
    def forward(self,x):
        x = x.view(-1,28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()


# In[76]:


f = MyNeuralNet()


# Look at network predictions (before optimization):

# In[77]:


xs.shape


# In[78]:


f(xs)


# In[79]:


ys


# In[80]:


L(f(xs), ys)


# We want these predictions f(xs) to match the ys for all images. For these to match, the loss function L(f(xs), ys) should be as small as possible. As such, we adjust the weights of f such that L becomes as small as possible. This is done below:

# # Training

# This training loop is copied from the previous tutorial, with a few modifications

# In[81]:


def train_model(dl, f, n_epochs=20):
    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    # Train model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            # Update the weights of the network
            opt.zero_grad() 
            loss_value = L(f(x), y) 
            loss_value.backward() 
            opt.step() 
            # Store training data
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)


# In[82]:


epoch_data, loss_data = train_model(train_dl, f)


# In[83]:


plt.plot(epoch_data, loss_data)
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Cross Entropy (per batch)')


# In[84]:


epoch_data_avgd = epoch_data.reshape(20,-1).mean(axis=1)
loss_data_avgd = loss_data.reshape(20,-1).mean(axis=1)


# In[85]:


plt.plot(epoch_data_avgd, loss_data_avgd, 'o--')
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Cross Entropy (avgd per epoch)')


# In[86]:


y_sample = train_ds[0][1]
y_sample


# In[87]:


x_sample = train_ds[0][0]
yhat_sample = f(x_sample)
yhat_sample


# In[88]:


torch.argmax(yhat_sample)


# In[89]:


plt.imshow(x_sample)


# In[90]:


xs, ys = train_ds[0:2000]
yhats = f(xs).argmax(axis=1)


# In[91]:


fig, ax = plt.subplots(10,4,figsize=(10,15))
for i in range(40):
    plt.subplot(10,4,i+1)
    plt.imshow(xs[i])
    plt.title(f'Predicted Digit: {yhats[i]}')
fig.tight_layout()
plt.show()


# In[92]:


xs, ys = test_ds[:2000]
yhats = f(xs).argmax(axis=1)


# In[93]:


fig, ax = plt.subplots(10,4,figsize=(10,15))
for i in range(40):
    plt.subplot(10,4,i+1)
    plt.imshow(xs[i])
    plt.title(f'Predicted Digit: {yhats[i]}')
fig.tight_layout()
plt.show()


# In[ ]:




