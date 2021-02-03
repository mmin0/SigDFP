

from data import data_generator
import src.model as model
from example import Invest
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

dic = {"in_dim": 5, "out_dim":1, "neurons":[64, 32, 32, 16]}
train_args = argparse.Namespace(**dic)
params = {'N':N, 'T':T, 'device': device}
mode = Invest.Invest(params) #specify the example

    
delta = torch.rand((B, 1))/2+5
mu = torch.rand((B, 1))/4+0.1
nu = torch.rand((B, 1))/5+0.2
theta = torch.rand((B, 1))
sigma = torch.rand((B, 1))/5 +0.2
typeVec = torch.cat([delta, mu, nu, theta, sigma], dim=1)

dat = torch.utils.data.TensorDataset(bm, w0, typeVec) 
DataLoader = torch.utils.data.DataLoader(dat, batch_size=512)

Alpha = model.Action1(train_args, mode)
criterion = model.LossTotal(mode, depth)

Alpha.load_state_dict(torch.load("Alpha1.net"))
Alpha.to(device)

mode.initialize(typeVec)
benchmark = mode.benchmark(bm, w0, initial)
benchmark_loss = mode.benchmark_loss(initial)
mode.distFlow(benchmark, rough_w0, depth=depth) # initialize linear regression object
mode.linear.coef_ = torch.load(os.path.join(params_path, "last_linear_coef_Invest.pt"))
mode.linear.intercept_ = torch.mean(initial).item()
benchmark_m = mode.Xbar
pi = mode.pi.clone()
m = mode.getDistFlow(B, w0, depth)

testLoss = utils.evaluate1(Alpha, DataLoader, criterion, initial,
                             m.to(device), device)
Alpha.to('cpu')
X = Alpha(bm, w0, typeVec, m, initial)

target_addr = sys.path[0]+'/plots/Invest'
f = open(target_addr+"/testout_Invest", "w")
sys.stdout = f
### test
print("-----------------------------")
print("----------Test Data----------")
print("-----------------------------")
print("Loss on test data: ", testLoss)

print("The L2 distance between controlled SDE on test data: ", 
          utils.L2distance(benchmark, X[:, :, 1:].cpu()))
print("The L2 (relative) distance between controlled SDE on test data: ", 
          utils.L2distance(benchmark, X[:, :, 1:].cpu())/utils.L2distance(benchmark, torch.zeros(B, N+1, 1)))

title = None
name = "SDE"
utils.plotSDE(benchmark[:3].cpu().detach().numpy(), X[:3, :, 1:].cpu().detach().numpy(),
                  target_addr, title, name, label1=r"$X_t$", label2=r"$\widehat{X}_t$")

utils.plotMeanDiff_bencmarkvspredicted([[i/N for i in range(N+1)], benchmark_m[:3, :].cpu().detach().numpy(),
                        m[:3, :].cpu().detach().numpy()],
                        target_addr, 
                        None, 'Xbar', label1=r"$m_t$", label2=r"$\widehat{m}_t$",
                        ylabel=r"$m_t$ and $\widehat{m}_t$")
print("The L2 distance between Xbar on test data: ", 
          utils.L2distance(benchmark_m.view(B, -1, 1), m.cpu().view(B, -1, 1)))
print("The L2 (relative) distance between Xbar on test data: ", 
          utils.L2distance(benchmark_m.view(B, -1, 1), m.cpu().view(B, -1, 1))/utils.L2distance(benchmark_m.view(B, -1, 1), torch.zeros(B, N+1, 1)))

pi_pred = torch.cat([Alpha.strategy[i].view(-1, 1) for i in range(N)], dim=1)
pi = torch.cat([pi for i in range(N)], dim=1)
title = None
name = "pi"
utils.plotpi(pi[:3].cpu().detach().numpy(), pi_pred[:3].cpu().detach().numpy(), 
                  target_addr, title, name, label1=r"$\pi_t$", label2=r"$\widehat{\pi}_t$", ylabel=r"$\pi_t$ and $\widehat{\pi}_t$", legendloc="best")

print("The L2 distance between pi on test data: ", 
          utils.L2distance(pi.view(B, -1, 1), pi_pred.cpu().view(B, -1, 1)))
print("The L2 (relative) distance between pi on test data: ", 
          utils.L2distance(pi.view(B, -1, 1), pi_pred.cpu().view(B, -1, 1))/utils.L2distance(pi.view(B, -1, 1), torch.zeros(B, N, 1)))

valid_utils = np.load(os.path.join(params_path, "valid_util_Invest.npy"))
utils.plotUtil(valid_utils, (0.8, 1.1), benchmark_loss, target_addr, None, "valid_util", 
               ins_loc=[0.55, 0.1, 0.25, 0.25], ins_ylim=(benchmark_loss-0.01, benchmark_loss+0.005))

f.close()
