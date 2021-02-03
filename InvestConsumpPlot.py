

from data import data_generator
import src.model as model
from example import InvestConsumption
import argparse
import torch
import src.utils as utils
import signatory
import sys
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, help='signature depth', required=True)
args = parser.parse_args()

torch.manual_seed(21)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
augment = signatory.Augment(1, 
                            layer_sizes = (), 
                            kernel_size = 1,
                            include_time = True)


params_path = os.path.join(sys.path[0], 'output')

B = 2**15
T = 1
N = 100 # mesh = 1/N
    

bm = data_generator.BMIncrements(B//2, N=N, T=T)
w_cn = data_generator.BMIncrements(B//2, N=N, T=T) #common noise
w0 = torch.zeros(B, N+1, 1)
for i in range(1, N+1):
    w0[:, i] = w0[:, i-1] + w_cn[:, i-1]

initial_generator = torch.rand
initial = initial_generator(B, 1)

depth = args.depth
rough_w0 = signatory.Path(augment(w0), depth, basepoint=False) # rough path

dic = {"in_dim": 6, "out_dim":2, "neurons":[64, 64, 64]}
train_args = argparse.Namespace(**dic)
params = {'N':N, 'T':T, 'device': device}
mode = InvestConsumption.InvestConsumption(params) #specify the example

delta = torch.rand((B, 1))/2+2
mu = torch.rand((B, 1))/4+0.1
nu = torch.rand((B, 1))/5 + 0.2
theta = torch.rand((B, 1))
sigma = torch.rand((B, 1))/5 +0.2
eps = torch.rand((B, 1))/2 + 0.5
typeVec = torch.cat([delta, mu, nu, theta, sigma, eps], dim=1)

dat = torch.utils.data.TensorDataset(bm, w0, typeVec) 
DataLoader = torch.utils.data.DataLoader(dat, batch_size=2**13)

Alpha = model.Action2(train_args, mode)
criterion = model.LossTotal2(mode, depth)



mode.initialize(typeVec)
benchmark, benchmark_c = mode.benchmark(bm, w0, initial)
benchmark_loss = mode.benchmark_loss
pi = mode.pi.clone()
benchmark_m = mode.Xbar.clone()
benchmark_mc = torch.exp(torch.mean(torch.log(benchmark_c.clone()), dim=0))
mode.distFlow(benchmark, benchmark_c, rough_w0) #initialize linear regression object

mode.linear.coef_ = torch.load(os.path.join(params_path, "last_linear_coef_InvestConsumption.pt"))
mode.linear.intercept_ = -1.00586
mode.linearc.coef_ = torch.load(os.path.join(params_path, "last_linearc_coef_InvestConsumption.pt"))
mode.linearc.intercept_ = -0.66
m, mc = mode.getDistFlow(B, w0, depth)


target_addr = sys.path[0]+'/plots/InvestConsumption'
f = open(target_addr+"/testout_InvestConsumption", "w")
sys.stdout = f
### test
print("-----------------------------")
print("----------Test Data----------")
print("-----------------------------")

Alpha.load_state_dict(torch.load('Alpha2.net'))
Alpha.to(device)
##
testLoss = utils.evaluate2(Alpha, DataLoader, criterion, 
                              initial, m.to(device), 
                              mc.to(device), device)
    
Alpha.to('cpu')
X = Alpha(bm, w0, typeVec, m, mc, initial)
    
print("Loss on test data: ", testLoss)
print("The L2 distance between controlled SDE on test data: ", 
          utils.L2distance(benchmark, X[:, :, 1:].cpu()))
print("The L2 (relative) distance between controlled SDE on test data: ", 
          utils.L2distance(benchmark, X[:, :, 1:].cpu())/utils.L2distance(benchmark, torch.zeros(B, N+1, 1)))
title = None
name = "SDE"
utils.plotSDE(benchmark[8:11].cpu().detach().numpy(), X[8:11, :, 1:].cpu().detach().numpy(),
                  target_addr, title, name, label1=r"$X_t$", label2=r"$\widehat{X}_t$", legendloc="upper center")

#utils.plotSDE_CI(benchmark.cpu().detach().numpy(), X[:, :, 1:].cpu().detach().numpy(),
#                  target_addr, title, name)
    
ctest = torch.cat([Alpha.strategy[i][:, 1:] for i in range(N)], dim=1)
ptest = torch.cat([Alpha.strategy[i][:, :1] for i in range(N)], dim=1)
    
print("The L2 distance between cost c on test data: ", 
          utils.L2distance(benchmark_c, ctest.cpu().view(B, -1, 1)))
print("The L2 (relative) distance between cost c on test data: ", 
          utils.L2distance(benchmark_c, ctest.cpu().view(B, -1, 1))/utils.L2distance(benchmark_c, torch.zeros(B, N, 1)))
          #/utils.L2distance(benchmark_test_c, torch.zeros(benchmark_test_c.size()))
title = None
utils.plotC(benchmark_c[8:11].cpu().detach().numpy(), ctest[8:11].cpu().detach().numpy(), 
                  target_addr, title, "c", label1 = r"$c_t$", label2=r"$\widehat{c}_t$",
                  ylabel=r"$c_t$ and $\widehat{c}_t$")
    
title = None
name = "Gamma"
utils.plotmC(benchmark_mc.view(N, -1).cpu().detach().numpy(), mc[8:11].cpu().detach().numpy(), 
                  target_addr, None, name, label1 = r"$\Gamma_t$", label2=r"$\widehat{\Gamma}_t$",
                  ylabel=r"$\Gamma_t$ and $\widehat{\Gamma}_t$")

gm = torch.cat([benchmark_mc.cpu().detach().view(1, N, -1) for i in range(B)], dim=0)
print("The L2 distance between gamma on test data: ", 
          utils.L2distance(gm, mc.cpu().view(B, -1, 1)))
print("The L2 (relative) distance between gamma on test data: ", 
          utils.L2distance(gm, mc.cpu().view(B, -1, 1))/utils.L2distance(gm, torch.zeros(B, N, 1)))
          
    
    
#title = "Xbar_t on test data"
#utils.plotmC(benchmark_test_mc.view(N, -1).cpu().detach().numpy(), mc_test[:5].cpu().detach().numpy(), 
#              target_addr, title, name)
utils.plotMeanDiff_bencmarkvspredicted([[i/N for i in range(N+1)], benchmark_m[8:11, :].cpu().detach().numpy(),
                        m[8:11, :].cpu().detach().numpy()],
                        target_addr, 
                        None, 'Xbar', ylim=(None, 0.50),label1=r"$m_t$", label2=r"$\widehat{m}_t$",
                        ylabel=r"$m_t$ and $\widehat{m}_t$", legendloc="upper center")
print("The L2 distance between Xbar on test data: ", 
          utils.L2distance(benchmark_m.view(B, -1, 1), m.cpu().view(B, -1, 1)))
print("The L2 (relative) distance between Xbar on test data: ", 
          utils.L2distance(benchmark_m.view(B, -1, 1), m.cpu().view(B, -1, 1))/utils.L2distance(benchmark_m.view(B, -1, 1), torch.zeros(B, N+1, 1)))
    
    
pi = torch.cat([pi for i in range(N)], dim=1)
print("The L2 distance between pi on test data: ", 
          utils.L2distance(pi.view(B, -1, 1), ptest.cpu().view(B, -1, 1)))
print("The L2 (relative) distance between pi on test data: ", 
          utils.L2distance(pi.view(B, -1, 1), ptest.cpu().view(B, -1, 1))/utils.L2distance(pi.view(B, -1, 1), torch.zeros(B, N, 1)))
title = None
name = "pi"
utils.plotpi(pi[8:11].cpu().detach().numpy(), ptest[8:11].cpu().detach().numpy(), 
                  target_addr, title, name, label1=r"$\pi_t$", label2=r"$\widehat{\pi}_t$", 
                  ylabel=r"$\pi_t$ and $\widehat{\pi}_t$", legendloc="best")

valid_utils = np.load(os.path.join(params_path, "valid_util_InvestConsumption.npy"))
utils.plotUtil(valid_utils, (2.0, 3.0), benchmark_loss, target_addr, None, "valid_util", 
               ins_loc=[0.55, 0.1, 0.25, 0.25], ins_ylim=(benchmark_loss-0.01, benchmark_loss+0.03))
#### also plot test loss in this plot.
f.close()
    

    


