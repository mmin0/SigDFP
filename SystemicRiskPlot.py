

from data import data_generator
import src.model as model
from example import SystemicRisk
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
bms = torch.utils.data.TensorDataset(bm, w0) 
bmDataLoader = torch.utils.data.DataLoader(bms, batch_size=2**10)

initial_generator = torch.rand
initial = initial_generator(B, 1)

depth = args.depth
rough_w0 = signatory.Path(augment(w0), depth, basepoint=False)


dic = {"in_dim": 2, "out_dim":1, "neurons":[64, 32]}
train_args = argparse.Namespace(**dic)
params = {'sigma':0.2, 'q':1, 'a':1, 'eps':1.5, 'rho':0.2, 'N':N, 'T':T, 'c':1, 'device': device}
mode = SystemicRisk.SystemicRisk(params)

Alpha = model.Action(train_args, mode)
criterion = model.LossTotal(mode, depth)
Alpha.load_state_dict(torch.load('Alpha.net'))
Alpha.to(device)
criterion.to(device)

linear_coef = torch.load(os.path.join(params_path, "last_linear_coef_SystemicRisk.pt"))
benchmark = mode.benchmark(bm, w0, initial)
benchmark_loss = mode.benchmark_loss(initial)
benchmark_alpha = mode.alpha
mode.distFlow(benchmark, rough_w0, depth=depth) # initialize linear regression object
mode.linear.coef_ = linear_coef
mode.linear.intercept_ = torch.mean(initial).item()
m = mode.getDistFlow(B, w0, depth)

testloss = utils.evaluate(Alpha, bmDataLoader, 
                                    criterion, initial,
                                    m.to(device), device)

Alpha.to('cpu')
X = Alpha(bm, w0, m.to('cpu'), initial)
predicted_alpha = torch.cat(Alpha.strategy, dim=1).cpu().detach()

target_addr = sys.path[0]+'/plots/SystemicRisk'
f = open(target_addr+"/testout_SystemicRisk", "w")
sys.stdout = f
print("-----------------------------")
print("----------Test Data----------")
print("-----------------------------")

print("Loss on test data: ", testloss)
print("The L2 distance between controlled SDE on test data: ", utils.L2distance(benchmark, X[:, :, 1:].cpu()))
print("The L2 (relative) distance between controlled SDE on test data: ", 
          utils.L2distance(benchmark, X[:, :, 1:].cpu())/utils.L2distance(benchmark, torch.zeros(B, N+1, 1)))

title = None #"Controlled SDE: Benchmark vs. Predicted"
name = "SDE"
utils.plotSDE(benchmark[:3].cpu().detach().numpy(), X[:3, :, 1:].cpu().detach().numpy(),
                  target_addr, title, name,
                  label1 = r"$X_t$", label2 = r"$\hat{X}_t$")

benchmark_m = (torch.mean(initial)+w0*params["sigma"]*params["rho"]).view(B, -1, 1)
print("The L2 distance between Xbar on test data: ",
          utils.L2distance(benchmark_m, m.cpu().view(B, -1, 1)))
print("The L2 (relative) distance between Xbar on test data: ", 
          utils.L2distance(benchmark_m, m.cpu().view(B, -1, 1))/utils.L2distance(benchmark_m, torch.zeros(B, N+1, 1)))

utils.plotMeanDiff_bencmarkvspredicted([[i/N for i in range(N+1)], m[:3, :].cpu().detach().numpy(),
                        benchmark_m[:3]],
                        target_addr, None, 'mt', ylim=(0.45, 0.6),
                        label1 = r"$m_t$" , label2 = r"$\hat{m}_t$", ylabel=r"$m_t$")

errors = np.load(os.path.join(params_path, "errors_SystemicRisk.npy"))
valid_util = np.load(os.path.join(params_path, "valid_util_SystemicRisk.npy"))
#utils.plotErrors(errors, target_addr, "", "")
utils.plotUtil(valid_util, (0, 0.05), benchmark_loss, target_addr, "Validation Cost", "valid_cost")

print("The L2 distance between pi on test data: ", 
          utils.L2distance(benchmark_alpha.view(B, -1, 1), predicted_alpha.cpu().view(B, -1, 1)))
print("The L2 (relative) distance between pi on test data: ", 
          utils.L2distance(benchmark_alpha.view(B, -1, 1), predicted_alpha.cpu().view(B, -1, 1))/utils.L2distance(benchmark_alpha.view(B, -1, 1), torch.zeros(B, N, 1)))

utils.plotpi(benchmark_alpha[:3], predicted_alpha[:3], target_addr, None, #r"$\alpha_t$: Benchmark vs. Predicted", 
             "alpha", label1 = r"$\alpha_t$", label2 = r"$\hat{\alpha}_t$", ylabel=r"$\alpha_t$")
f.close()






