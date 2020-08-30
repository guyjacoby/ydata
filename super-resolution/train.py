import numpy as np
import torch

from tqdm.notebook import tqdm
from math import log10

def train_eval(train_loader, val_loader, net, criterion, optimizer, num_epochs):
    
    training_log = []
    f = np.vectorize(log10)
    
    if torch.has_cuda:
        device = torch.device('cuda:0')
        net = net.to(device)
    else:
        device = torch.device('cpu:0')

    for epoch in range(num_epochs):

        net.train()
        train_loss = 0

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
            inputs, targets_mid, targets_large = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets_mid).mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch+1, 
                                                                 train_loss / len(train_loader)))
        
        net.eval()
        val_loss = 0
        psnr_sum = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='validation'):
                inputs, targets_mid, targets_large = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets_mid)
                loss = loss.view(inputs.shape[0],-1).mean(1)
                val_loss += loss.sum().item()
                psnr_sum += (10*f(1 / loss.cpu())).sum()           
    
        print("===> Avg. PSNR: {:.4f} dB".format(psnr_sum / (len(val_loader)*inputs.shape[0])))
            
        training_log.append({
            'epoch':epoch+1,
            'train_loss':train_loss / len(train_loader),
            'validation_loss':val_loss / (len(val_loader)*inputs.shape[0]),
            'psnr_mid':psnr_sum / (len(val_loader)*inputs.shape[0])
        })
    return training_log

def train_eval_two_outputs(train_loader, val_loader, net, criterion, optimizer, num_epochs):
    
    training_log = []
    f = np.vectorize(log10)
    
    if torch.has_cuda:
        device = torch.device('cuda:0')
        net = net.to(device)
    else:
        device = torch.device('cpu:0')

    for epoch in range(num_epochs):

        net.train()
        train_loss = 0

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
            inputs, targets_mid, targets_large = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            outputs_mid, outputs_large = net(inputs)
            loss_mid = criterion(outputs_mid, targets_mid).mean()
            loss_large = criterion(outputs_large, targets_large).mean()
            loss = loss_mid + loss_large
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch+1, 
                                                                 train_loss / len(train_loader)))
        
        net.eval()
        val_loss = 0
        psnr_mid = 0
        psnr_large = 0
        
        with torch.no_grad():
            for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='validation'):
                inputs, targets_mid, targets_large = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                outputs_mid, outputs_large = net(inputs)
                loss_mid = criterion(outputs_mid, targets_mid)
                loss_mid = loss_mid.view(inputs.shape[0],-1).mean(1)
                loss_large = criterion(outputs_large, targets_large)
                loss_large = loss_large.view(inputs.shape[0],-1).mean(1)
                val_loss += loss_mid.sum().item() + loss_large.sum().item()
                psnr_mid += (10*f(1 / loss_mid.cpu())).sum()
                psnr_large += (10*f(1 / loss_large.cpu())).sum()
    
        print("===> Avg. PSNR-Mid: {:.4f} dB, Avg. PSNR-Large: {:.4f} dB".format(psnr_mid / (len(val_loader)*inputs.shape[0]),
                                                                                 psnr_large / (len(val_loader)*inputs.shape[0])))
            
        training_log.append({
            'epoch':epoch+1,
            'train_loss':train_loss / len(train_loader),
            'validation_loss':val_loss / (len(val_loader)*inputs.shape[0]),
            'psnr_mid':psnr_mid / (len(val_loader)*inputs.shape[0]),
            'psnr_large':psnr_large / (len(val_loader)*inputs.shape[0])
        })
    return training_log