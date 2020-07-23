import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_log(log):
    
    Epochs = np.arange(len(log))+1
    train_MSE = [i['train_loss'] for i in log]
    validation_MSE = [i['validation_loss'] for i in log]
    PSNR_Mid = [i['psnr_mid'] for i in log]
    if 'psnr_large' in log[0].keys():
        PSNR_Large = [i['psnr_large'] for i in log]
    
    plt.figure(0, figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(Epochs, train_MSE, '-*b', label='train')
    plt.plot(Epochs, validation_MSE, '-*r', label='validation')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.xticks(Epochs.tolist(), fontsize=14)
    plt.yticks(fontsize=14)
    ax = plt.gca()
    ax.set(xticks=np.arange(10,101, 10), xticklabels=np.arange(10,101, 10))
    plt.legend(fontsize=14)
    plt.subplot(1,2,2)
    plt.plot(Epochs, PSNR_Mid, '-*b', label='mid')
    if 'psnr_large' in log[0].keys():
        plt.plot(Epochs, PSNR_Large, '-*r', label='large')
    plt.ylabel('PSNR (dB)', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.xticks(Epochs.tolist(), fontsize=14)
    plt.yticks(fontsize=14)
    ax = plt.gca()
    ax.set(xticks=np.arange(10,101, 10), xticklabels=np.arange(10,101, 10))
    plt.legend(fontsize=14)
    plt.show()

def plot_samples(model, valloader, device='cpu:0'):
    images = iter(valloader).next()
    X, y = images[0], images[1:]
    model = model.to(device)
    model.eval()
    outputs = model(X.to(device))

    if isinstance(outputs,torch.Tensor):
        outputs = [outputs]

    fig, axes = plt.subplots(nrows=4, ncols=len(outputs) * 2 + 1, figsize=(18, 18))

    cols = ['X', 'Target 144x144', 'Output 144x144', 'Target 288x288', 'Output 288x288']

    for i, axs in enumerate(axes):
        axs[0].imshow(X[i,:,:,:].permute(1,2,0))
        
        for j, out in enumerate(outputs):
            axs[2*j + 1].imshow(y[j][i,:,:,:].permute(1,2,0))
            axs[2*j + 2].imshow(out[i,:,:,:].permute(1,2,0).cpu().detach())

        if i == 0:
            for ax, col in zip(axs, cols):
                ax.set_title(col, fontsize=16)