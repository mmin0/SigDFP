

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FormatStrFormatter
#from tqdm import tqdm

plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
plt.rc('ytick', labelsize=22)
plt.rc('legend', fontsize=25) 
plt.rc('axes', labelsize=25)
plt.rcParams["figure.figsize"] = (7.5, 6)
colors = ['lightcoral', 'mediumseagreen', 'darkorange']

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
        i+=1
    return epoch_loss/len(dataloader)



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
        
        i+=1
    return epoch_loss/len(dataloader)




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
        
        i+=1
    return epoch_loss/len(dataloader)


def plotErrors(error, target_addr, title, filename):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel("FP rounds")
    plt.ylabel("Errors")
    plt.plot(error, color='blue')
    fig.savefig(target_addr+'/'+filename+'.pdf')
    
def plotUtil(util, ylim, ytrue, target_addr, title, filename, ins_loc=None, ins_ylim=None, cost=False):
    
    if cost:
        n = len(util)
        fig, ax = plt.subplots(figsize=(7.5, 6))
        if title:
            plt.title(title)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel(r"FP iterations $n$")
        ax.set_ylabel("validation cost")
        l1 = ax.axhline(ytrue, color="indianred", ls="--")
        l2, = ax.plot(util, color='darkcyan', ls="-")
        if ins_loc:
            axins = ax.inset_axes(ins_loc)
        if ins_ylim:
            axins.set_ylim(ins_ylim)
        axins.plot(range(n-50, n), util[-50:], color='darkcyan', ls="-")
        axins.axhline(ytrue, color="indianred", ls="--")
        
        ax.indicate_inset_zoom(axins)
        ax.legend((l1, l2), ("true cost", "validation cost"), loc="upper center")
        plt.tight_layout()
        fig.savefig(target_addr+'/'+filename+'.pdf')
    else:
        n = len(util)
        fig, ax = plt.subplots(figsize=(7.5, 6))
        if title:
            plt.title(title)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel(r"FP iterations $n$")
        ax.set_ylabel("validation utility")
        l1 = ax.axhline(ytrue, color="indianred", ls="--")
        l2, = ax.plot(util, color='darkcyan', ls="-")
        if ins_loc:
            axins = ax.inset_axes(ins_loc)
        if ins_ylim:
            axins.set_ylim(ins_ylim)
        axins.plot(range(n-50, n), util[-50:], color='darkcyan', ls="-")
        axins.axhline(ytrue, color="indianred", ls="--")
        
        ax.indicate_inset_zoom(axins)
        ax.legend((l1, l2), ("true utility", "validation utility"), loc="upper center")
        plt.tight_layout()
        fig.savefig(target_addr+'/'+filename+'.pdf')



def plotMeanDiff_bencmarkvspredicted(data, target_addr, title, filename, ylim=None, label1=None, label2=None, ylabel=None, legendloc=None, round_=False):
    fig = plt.figure()
    if title:
        plt.title(title)
    x, next_m, m = data
    if ylim:
        plt.ylim(ylim)
    c = len(next_m)
    lines = []
    lines_pred = []
    for i in range(c):
        l, = plt.plot(x, next_m[i], color=colors[i])
        lines.append(l)
    for i in range(c):
        l, = plt.plot(x, m[i], color=colors[i], ls='--', marker='.')
        lines_pred.append(l)
    #plt.plot(x, next_m-m, label='diff')
    plt.xlabel(r"time $t$")
    if ylabel:
        plt.ylabel(ylabel)
    if legendloc:
        plt.legend([tuple(lines), tuple(lines_pred)], [label1, label2],
                loc=legendloc, ncol=2, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        plt.legend([tuple(lines), tuple(lines_pred)], [label1, label2],
                loc="upper left", ncol=2, handler_map={tuple: HandlerTuple(ndivide=None)})
    if round_:
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()
    fig.savefig(target_addr+'/'+filename+'.pdf')
    

def L2distance(x, y):
    b, N, _ = x.size()
    return ((torch.sum(torch.pow(x - y, 2))/N/b)**0.5).item()

def plotSDE(benchmark, predicted, target_addr, title, filename, ylim=None, label1=None, label2=None, legendloc=None):
    """
    input:
        benchmark -- list[paths]
        predicted -- list[paths]
    """
    fig = plt.figure()
    if title:
        plt.title(title)
    if ylim:
        plt.ylim(ylim)
    t = [i/100 for i in range(101)]
    c = len(benchmark)
    lines = []
    lines_pred = []
    for i in range(c):
        l, = plt.plot(t, benchmark[i], color=colors[i], ls='-')
        lines.append(l)
    for i in range(c):
        l, = plt.plot(t, predicted[i], color=colors[i], ls='--', marker='.')
        lines_pred.append(l)
    if legendloc:
        plt.legend([tuple(lines), tuple(lines_pred)], [label1, label2],
                loc=legendloc, ncol=2, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        plt.legend([tuple(lines), tuple(lines_pred)], [label1, label2],
                loc="upper left", ncol=2, handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.xlabel(r"time $t$")
    plt.ylabel(r"$X_t$ and $\widehat{X}_t$")
    plt.tight_layout()
    fig.savefig(target_addr+'/'+filename+'.pdf')
    

    
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
    c = len(benchmark)
    lines = []
    lines_pred = []
    for i in range(c):
        l, = plt.plot(t, benchmark[i], color=colors[i], ls='-')
        lines.append(l)
    for i in range(c):
        l, = plt.plot(t, predicted[i], color=colors[i], ls='--', marker='.')
        lines_pred.append(l)
    plt.legend([tuple(lines), tuple(lines_pred)], [label1, label2], 
                loc="upper left",ncol=2, handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.xlabel(r"time $t$")
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(target_addr+'/'+filename+'.pdf')
    
def plotpi(benchmark, predicted, target_addr, title, filename, ylim = None, label1=None, label2=None, ylabel=None, legendloc = None):
    """
    input:
        benchmark -- list[paths]
        predicted -- list[paths]
    """
    fig = plt.figure()
    if title:
        plt.title(title)
    if ylim:
        plt.ylim(ylim)
    t = [i/100 for i in range(100)]
    c = len(benchmark)
    lines = []
    lines_pred = []
    for i in range(c):
        l, = plt.plot(t, benchmark[i], color=colors[i], ls='-')
        lines.append(l)
    for i in range(c):
        l, = plt.plot(t, predicted[i], color=colors[i], ls='--', marker='.')
        lines_pred.append(l)
    if legendloc:
        plt.legend([tuple(lines), tuple(lines_pred)], [label1, label2],
                loc=legendloc, ncol=2, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        plt.legend([tuple(lines), tuple(lines_pred)], [label1, label2],
                loc="upper left", ncol=2, handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.xlabel(r"time $t$")
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(target_addr+'/'+filename+'.pdf')
    
    
def plotmC(benchmark, predicted, target_addr, title, filename, label1=None, label2=None, ylabel=None):
    """
    input:
        benchmark -- list[paths]
        predicted -- list[paths]
    """
    N = predicted.shape[1]
    t = [1/N*i for i in range(N)]
    fig = plt.figure()
    if title:
        plt.title(title)
    
    c = len(predicted)
    lines = []
    lines_pred = []
    l, = plt.plot(t, benchmark, color='darkgrey', ls='-', linewidth=5)
    lines.append(l)
    for i in range(c):
        l, = plt.plot(t, predicted[i], color=colors[i], ls='--', marker='.')
        lines_pred.append(l)
    plt.legend([tuple(lines), tuple(lines_pred)], [label1, label2], 
                loc="upper left", ncol=2, handler_map={tuple: HandlerTuple(ndivide=None)})
    
    plt.xlabel(r"time $t$")
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(target_addr+'/'+filename+'.pdf')
    
