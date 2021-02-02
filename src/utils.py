

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#from tqdm import tqdm

plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=20) 
plt.rc('axes', labelsize=20)

def epoch_time(start_time, end_time):
    elap_time = end_time - start_time
    elap_min = elap_time//60
    elap_sec = elap_time % 60
    return elap_min, elap_sec


def train(model, dataloader, optimizer, criterion, initial, prev_m, device, depth=4):
    """
    train model for alpha for one loop over dataloader
    """
    epoch_loss = 0
    
    model.train() # set model to train mode
    i = 0
    for batch in dataloader:
        optimizer.zero_grad()
    
        bm, cn = batch
        
        X = model(bm.to(device), 
                  cn.to(device), 
                  prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size], 
                  initial[i*dataloader.batch_size:(i+1)*dataloader.batch_size].to(device))
        strategy = model.strategy
        loss = criterion(X, prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size], strategy)
        
        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss += loss.item()
        i+=1
    return epoch_loss/len(dataloader)


def evaluate(model, dataloader, criterion, initial, prev_m, device, depth=4):
    
    epoch_loss = 0
    #initialize
    #N = prev_m.size()[0] - 1
    #m = torch.zeros(N+1, 1, device=device)
    #sigs = torch.zeros(signatory.signature_channels(2, depth), device=device)
    model.eval() # set model to train mode
    i = 0
    for batch in dataloader:
        bm, cn = batch
        X = model(bm.to(device), 
                  cn.to(device), 
                  prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size], 
                  initial[i*dataloader.batch_size:(i+1)*dataloader.batch_size].to(device))
        strategy = model.strategy
        loss = criterion(X, prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size], strategy)

        epoch_loss += loss.item()
        #m += torch.mean(X[:, :, 1:], dim=0)
        #sigs += torch.mean(signatory.signature(X, depth, basepoint=True), dim=0)
        i+=1
    return epoch_loss/len(dataloader)#, m/(len(dataloader)), sigs/(len(dataloader))



def train1(model, dataloader, optimizer, criterion, initial, prev_m, device, depth=4):
    """
    train model for alpha for one loop over dataloader
    """
    epoch_loss = 0
    
    model.train() # set model to train mode
    i = 0
    for batch in dataloader:
        
        optimizer.zero_grad()
    
        bm, cn, typeVec = batch
        
        X = model(bm.to(device), 
                  cn.to(device), 
                  typeVec.to(device),
                  prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size], 
                  initial[i*dataloader.batch_size:(i+1)*dataloader.batch_size].to(device))
        strategy = model.strategy
        loss = criterion(X, prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size], strategy)
        
        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss += loss.item()
        i+=1
    return epoch_loss/len(dataloader)


def evaluate1(model, dataloader, criterion, initial, prev_m, device, depth=4):
    
    epoch_loss = 0
    model.eval() # set model to train mode
    i = 0
    for batch in dataloader:
        
        bm, cn, typeVec = batch
        X = model(bm.to(device), 
                  cn.to(device),
                  typeVec.to(device),
                  prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size], 
                  initial[i*dataloader.batch_size:(i+1)*dataloader.batch_size].to(device))
        strategy = model.strategy
        loss = criterion(X, prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size], strategy)

        epoch_loss += loss.item()
        #m += torch.mean(X[:, :, 1:], dim=0)
        #sigs += torch.mean(signatory.signature(X, depth, basepoint=True), dim=0)
        i+=1
    return epoch_loss/len(dataloader)#, m/(len(dataloader)), sigs/(len(dataloader))




def train2(model, dataloader, optimizer, criterion, initial, prev_m, prev_c, device, depth=4):
    """
    train model for alpha for one loop over dataloader
    """
    epoch_loss = 0
    
    model.train() # set model to train mode
    i = 0
    for batch in dataloader:
        
        optimizer.zero_grad()
    
        bm, cn, typeVec = batch
        
        X = model(bm.to(device), 
                  cn.to(device), 
                  typeVec.to(device),
                  prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size],
                  prev_c[i*dataloader.batch_size:(i+1)*dataloader.batch_size],
                  initial[i*dataloader.batch_size:(i+1)*dataloader.batch_size].to(device))
        strategy = model.strategy
        
        loss = criterion(X, prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size], 
                        strategy, prev_c[i*dataloader.batch_size:(i+1)*dataloader.batch_size])
        
        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss += loss.item()
        i+=1
    return epoch_loss/len(dataloader)


def evaluate2(model, dataloader, criterion, initial, prev_m, prev_c, device, depth=4):
    
    epoch_loss = 0
    #initialize
    #N = prev_m.size()[0] - 1
    #m = torch.zeros(N+1, 1, device=device)
    #sigs = torch.zeros(signatory.signature_channels(2, depth), device=device)
    model.eval() # set model to train mode
    i = 0
    
    for batch in dataloader:
    
        bm, cn, typeVec = batch
        
        X = model(bm.to(device), 
                  cn.to(device), 
                  typeVec.to(device),
                  prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size], 
                  prev_c[i*dataloader.batch_size:(i+1)*dataloader.batch_size],
                  initial[i*dataloader.batch_size:(i+1)*dataloader.batch_size].to(device))
        strategy = model.strategy
        loss = criterion(X, prev_m[i*dataloader.batch_size:(i+1)*dataloader.batch_size], 
                        strategy, prev_c[i*dataloader.batch_size:(i+1)*dataloader.batch_size])
        
        epoch_loss += loss.item()
        #m += torch.mean(X[:, :, 1:], dim=0)
        #sigs += torch.mean(signatory.signature(X, depth, basepoint=True), dim=0)
        i+=1
    return epoch_loss/len(dataloader)#, m/(len(dataloader)), sigs/(len(dataloader))


def plotErrors(error, target_addr, title, filename):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel("FP rounds")
    plt.ylabel("Errors")
    plt.plot(error, color='blue')
    fig.savefig(target_addr+'/'+filename+'.pdf')
    
def plotUtil(util, ylim, ytrue, target_addr, title, filename):
    fig = plt.figure()
    if title:
        plt.title(title)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel("FP rounds")
    plt.ylabel("Utility")
    plt.axhline(ytrue, color="indianred", ls="-", label="true utility")
    plt.plot(util, color='darkcyan', ls="--", label="valid utility")
    plt.legend() #loc="center right"
    fig.savefig(target_addr+'/'+filename+'.pdf')




def plotMeanDiff(data, target_addr, title, filename):
    fig = plt.figure()
    plt.title(title)
    x, next_m, m = data
    for p in next_m:
        plt.plot(x, p, color='indianred', label='Dist Flow.')
    for p in m:
        plt.plot(x, p, color='grey', label='Previous Dist Flow', ls='--')
    #plt.plot(x, next_m-m, label='diff')
    plt.legend()
    #plt.ylim(-0.2, 0.2)
    fig.savefig(target_addr+'/'+filename+'.pdf')
    

def plotMeanDiff_bencmarkvspredicted(data, target_addr, title, filename, ylim=None, label1=None, label2=None, ylabel=None):
    fig = plt.figure()
    if title:
        plt.title(title)
    x, next_m, m = data
    if ylim:
        plt.ylim(ylim)
    for p in next_m:
        if label1:
            plt.plot(x, p, color='indianred', label=label1)
        else:
            plt.plot(x, p, color='indianred', label='Benchmark')
    for p in m:
        if label2:
            plt.plot(x, p, color='grey', label=label2, ls='--')
        else:
            plt.plot(x, p, color='grey', label='Predicted', ls='--')
    #plt.plot(x, next_m-m, label='diff')
    plt.xlabel(r"$t$")
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    #plt.ylim(-0.2, 0.2)
    fig.savefig(target_addr+'/'+filename+'.pdf')
    

def L2distance(x, y):
    b, N, _ = x.size()
    return ((torch.sum(torch.pow(x - y, 2))/N/b)**0.5).item()

def plotSDE(benchmark, predicted, target_addr, title, filename, label1=None, label2=None):
    """
    input:
        benchmark -- list[paths]
        predicted -- list[paths]
    """
    fig = plt.figure()
    if title:
        plt.title(title)
    t = [i/100 for i in range(101)]
    for p in benchmark:
        if label1:
            plt.plot(t, p, color='indianred', ls='-', label=label1)
        else:
            plt.plot(t, p, color='indianred', ls='-', label="benchmark")
    for p in predicted:
        if label2:
            plt.plot(t, p, color='grey', ls='--', label=label2)
        else:
            plt.plot(t, p, color='grey', ls='--', label="predicted SDE")
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$X_t$")
    fig.savefig(target_addr+'/'+filename+'.pdf')
    
def plotSDE_CI(benchmark, predicted, target_addr, title, filename):
    """
    input:
        benchmark -- array[paths]
        predicted -- array[paths]
    """
    fig = plt.figure()
    plt.title(title)
    N = benchmark.shape[1]
    t_axis = [1/N*i for i in range(N)]
    std = np.std(benchmark - predicted, axis=0).squeeze()
    for i in range(3):
        plt.plot(t_axis, benchmark[i], color='indianred', ls='-', label="benchmark")
    
    for i in range(3):
        plt.plot(t_axis, predicted[i], color='grey', ls='--', label="predicted SDE")
        plt.fill_between(t_axis, predicted[i].squeeze()-1.96*std, predicted[i].squeeze()+1.96*std, color='grey', alpha=0.1)
    plt.legend()
    fig.savefig(target_addr+'/'+filename+'SDE_CI.pdf')
    
def plotC(benchmark, predicted, target_addr, title, filename, label1=None, label2=None, ylabel=None):
    """
    input:
        benchmark -- list[paths]
        predicted -- list[paths]
    """
    t = [i/100 for i in range(100)]
    fig = plt.figure()
    if title:
        plt.title(title)
    for p in benchmark:
        if label1:
            plt.plot(t, p, color='indianred', ls='-', label=label1)
        else:
            plt.plot(t, p, color='indianred', ls='-', label="benchmark")
    for p in predicted:
        if label2:
            plt.plot(t, p, color='grey', ls='--', label=label2)
        else:
            plt.plot(t, p, color='grey', ls='--', label="predicted")
    plt.xlabel(r"$t$")
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    fig.savefig(target_addr+'/'+filename+'.pdf')
    
def plotpi(benchmark, predicted, target_addr, title, filename, label1=None, label2=None, ylabel=None):
    """
    input:
        benchmark -- list[paths]
        predicted -- list[paths]
    """
    fig = plt.figure()
    if title:
        plt.title(title)
    t = [i/100 for i in range(100)]
    for p in benchmark:
        if label1:
            plt.plot(t, p, color='indianred', ls='-', label=label1)
        else:
            plt.plot(t, p, color='indianred', ls='-', label="benchmark")
    for p in predicted:
        if label2:
            plt.plot(t, p, color='grey', ls='--', label=label2)
        else:
            plt.plot(t, p, color='grey', ls='--', label=predicted)
    plt.legend()
    plt.xlabel(r"$t$")
    if ylabel:
        plt.ylabel(ylabel)
    fig.savefig(target_addr+'/'+filename+'.pdf')
    
    
def plotmC(benchmark, predicted, target_addr, title, filename, label1=None, label2=None, ylabel=None):
    """
    input:
        benchmark -- list[paths]
        predicted -- list[paths]
    """
    N = predicted.shape[1]
    t_axis = [1/N*i for i in range(N)]
    fig = plt.figure()
    if title:
        plt.title(title)
    if label1:
        plt.plot(t_axis, benchmark, color='indianred', ls='-', label=label1)
    else:
        plt.plot(t_axis, benchmark, color='indianred', ls='-', label="benchmark")
    for p in predicted:
        if label2:
            plt.plot(t_axis, p, color='grey', ls='--', label=label2)
        else:
            plt.plot(t_axis, p, color='grey', ls='--', label="predicted")
    plt.xlabel(r"t")
    if ylabel:
        plt.ylabel(r"$\Gamma_t$")
    plt.legend()
    fig.savefig(target_addr+'/'+filename+'.pdf')
    
