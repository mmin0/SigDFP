
from data import data_generator
import src.model as model
from example import SystemicRisk
from example import Invest
from example import InvestConsumption
import argparse
import torch
import src.utils as utils
import time
import signatory
import torch.optim as optim
import sys
import numpy as np



torch.manual_seed(521)
case = "SystemicRisk" #SystemicRisk, Invest, InvestConsumption

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

augment = signatory.Augment(1, 
                            layer_sizes = (), 
                            kernel_size = 1,
                            include_time = True)


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

bm_test = data_generator.BMIncrements(B//2, N=N, T=T)
w_cn = data_generator.BMIncrements(B//2, N=N, T=T) #common noise
w0_test = torch.zeros(B, N+1, 1)
for i in range(1, N+1):
    w0_test[:, i] = w0_test[:, i-1] + w_cn[:, i-1]
bms_test = torch.utils.data.TensorDataset(bm_test, w0_test) 
bmDataLoader_test = torch.utils.data.DataLoader(bms_test, batch_size=2**12)


B_valid = B//2
bm_valid = data_generator.BMIncrements(B_valid//2, N=N, T=T)
w_cn = data_generator.BMIncrements(B_valid//2, N=N, T=T) #common noise
w0_valid = torch.zeros(B_valid, N+1, 1)
for i in range(1, N+1):
    w0_valid[:, i] = w0_valid[:, i-1] + w_cn[:, i-1]
bms_valid = torch.utils.data.TensorDataset(bm_valid, w0_valid) 
bmDataLoader_valid = torch.utils.data.DataLoader(bms_valid, batch_size=2**12)

initial_generator = torch.rand
initial = initial_generator(B, 1)#torch.randn(B, 1, device=device)
initial_valid = initial_generator(B_valid, 1)
initial_test = initial_generator(B, 1)



if case == 'SystemicRisk':
    depth = 2
    rough_w0 = signatory.Path(augment(w0), depth, basepoint=False) # rough path
    rough_w0_valid = signatory.Path(augment(w0_valid), depth, basepoint=False)
    print("Example: SystemicRisk")
    dic = {"in_dim": 2, "out_dim":1, "neurons":[64, 32]}
    train_args = argparse.Namespace(**dic)
    params = {'sigma':0.2, 'q':1, 'a':1, 'eps':1.5, 'rho':0.2, 'N':N, 'T':T, 'c':1, 'device': device}
    mode = SystemicRisk.SystemicRisk(params) #specify the example
    errors = []
    
    Alpha = model.Action(train_args, mode)
    criterion = model.LossTotal(mode, depth)
    optimizer = optim.SGD(Alpha.parameters(), lr=0.1)

    Alpha.to(device)
    criterion.to(device)

    m = torch.randn(B, N+1, 1, device=device)+2
    m_valid = torch.randn(B, N+1, 1, device=device)+2

    benchmark = mode.benchmark(bm, w0, initial)
    benchmark_test = mode.benchmark(bm_test, w0_test, initial_test)
    benchmark_loss = mode.benchmark_loss(initial_test)
    print("Benchmark loss: ", benchmark_loss)
    target_addr = sys.path[0]+'/output'

    # run training
    N_round = 500
    N_epoch = 1
    
    valid_util = []
    last_linear_coef = np.array([[0]*signatory.signature_channels(2, depth)])
    for r in range(N_round):
        optimizer.param_groups[0]['lr'] = .1/(10**(r//250))
        print("round: ", r+1)
        best_valid_loss = float('inf')
        for epoch in range(N_epoch):
            start_time = time.time()
        
            train_loss = utils.train(Alpha, bmDataLoader,
                                    optimizer, criterion, initial,
                                    m, device)
       
            valid_loss = utils.evaluate(Alpha, bmDataLoader_valid, 
                                    criterion, initial_valid,
                                    m_valid, device)
            end_time = time.time()
            epoch_min, epoch_sec = utils.epoch_time(start_time, end_time)

            if r == N_round-1 and valid_loss < best_valid_loss:
                print("save model")
                best_valid_loss = valid_loss
                torch.save(Alpha.state_dict(), 'Alpha.net')
        
            print(f'Epoch: {epoch+1} | Epoch Time: {epoch_min}m {epoch_sec:.2f}s')
            print(f'\tTrain Loss: {train_loss:.5f}')
            print(f'\t Val. Loss: {valid_loss:.5f}')
        Alpha.to('cpu')
        X = Alpha(bm, w0, m.to('cpu'), initial)
    
        ####################################################################
        # Average distribution
        mode.distFlow(X[:, :, 1:], rough_w0, depth=depth)
        if r>=250:
            mode.linear.coef_ = last_linear_coef* ((r-250)/(r-250+1)) + mode.linear.coef_/(r-250+1)
        last_linear_coef = mode.linear.coef_
        
        next_m = mode.getDistFlow(B, w0, depth)
        m_valid = mode.getDistFlow(B_valid, w0_valid, depth)
        ####################################################################
        valid_util.append(valid_loss)
    
        m = next_m.detach().to(device)
        m_valid = m_valid.to(device)
        e = utils.L2distance(benchmark, X[:, :, 1:].cpu())
        errors.append(e)
        print("The L2 distance between controlled SDE is: ", e)
        Alpha.to(device)

    #utils.plotErrors(errors, target_addr, "L2 Errors", "L2 Errors")
    #utils.plotMeanDiff_bencmarkvspredicted([[i/N for i in range(N+1)], m[:2, :].cpu().detach().numpy(),
    #                     torch.mean(initial)+w0[:2]*params["sigma"]*params["rho"]],
    #                    target_addr, 'Predict vs. Benchmark', 'benchmark')

    print("The linear functional: " +str(mode.linear.intercept_)+"," +str(mode.linear.coef_))
    torch.save(last_linear_coef, target_addr+"/last_linear_coef_SystemicRisk.pt")
    with open(target_addr + '/valid_util_SystemicRisk.npy', 'wb') as f:
        np.save(f, np.array(valid_util))
    with open(target_addr + '/errors_SystemicRisk.npy', 'wb') as f:
        np.save(f, np.array(errors))
    ### test
    
    Alpha.load_state_dict(torch.load('Alpha.net'))
    f = open(target_addr+"/testout", "w")
    sys.stdout = f
    
    print("-----------------------------")
    print("----------Test Data----------")
    print("-----------------------------")
    
    m_test = mode.getDistFlow(B, w0_test, depth)
    X = Alpha(bm_test.to(device), w0_test.to(device), m_test.to(device), initial_test.to(device))
    testLoss = utils.evaluate(Alpha, bmDataLoader_test, criterion, initial_test, m_test.to(device), device)
    print("Loss on test data: ", testLoss)
    print("The L2 distance between controlled SDE on test data: ", utils.L2distance(benchmark_test, X[:, :, 1:].cpu()))
    title = "Controlled SDE on test data"
    name = "Test result"
    utils.plotSDE(benchmark_test[:2].cpu().detach().numpy(), X[:2, :, 1:].cpu().detach().numpy(),
                  target_addr, title, name)
    print("The L2 distance between Xbar on test data: ",
          utils.L2distance((torch.mean(initial_test)+w0_test*params["sigma"]*params["rho"]).view(B, -1, 1), m_test.cpu().view(B, -1, 1)))
    utils.plotMeanDiff_bencmarkvspredicted([[i/N for i in range(N+1)], m_test[:2, :].cpu().detach().numpy(),
                        torch.mean(initial_test)+w0_test[:2]*params["sigma"]*params["rho"]],
                        target_addr, 'Predict vs. Benchmark on test data', 'benchmark_test')
    f.close()


if case == 'Invest':
    depth = 2
    rough_w0 = signatory.Path(augment(w0), depth, basepoint=False) # rough path
    rough_w0_valid = signatory.Path(augment(w0_valid), depth, basepoint=False)
    print("Example: Invest")
    dic = {"in_dim": 5, "out_dim":1, "neurons":[64, 32, 32, 16]}
    train_args = argparse.Namespace(**dic)
    params = {'N':N, 'T':T, 'device': device}
    mode = Invest.Invest(params) #specify the example
    errors = []
    
    delta = torch.rand((B, 1))/2+5
    mu = torch.rand((B, 1))/4+0.1
    nu = torch.rand((B, 1))/5+0.2
    theta = torch.rand((B, 1))
    sigma = torch.rand((B, 1))/5 +0.2
    typeVec = torch.cat([delta, mu, nu, theta, sigma], dim=1)
    
    delta = torch.rand((B, 1))/2+5
    mu = torch.rand((B, 1))/4+0.1
    nu = torch.rand((B, 1))/5+0.2
    theta = torch.rand((B, 1))
    sigma = torch.rand((B, 1))/5 +0.2
    typeVec_test = torch.cat([delta, mu, nu, theta, sigma], dim=1)
    
    
    delta = torch.rand((B_valid, 1))/2+5
    mu = torch.rand((B_valid, 1))/4+0.1
    nu = torch.rand((B_valid, 1))/5+0.2
    theta = torch.rand((B_valid, 1))
    sigma = torch.rand((B_valid, 1))/5 +0.2
    typeVec_valid = torch.cat([delta, mu, nu, theta, sigma], dim=1)
    
    dat = torch.utils.data.TensorDataset(bm, w0, typeVec) 
    train_DataLoader = torch.utils.data.DataLoader(dat, batch_size=2**10)
    dat = torch.utils.data.TensorDataset(bm_test, w0_test, typeVec_test) 
    test_DataLoader = torch.utils.data.DataLoader(dat, batch_size=2**13)
    dat = torch.utils.data.TensorDataset(bm_valid, w0_valid, typeVec_valid) 
    valid_DataLoader = torch.utils.data.DataLoader(dat, batch_size=2**13)
    
    Alpha = model.Action1(train_args, mode)
    criterion = model.LossTotal(mode, depth)
    optimizer = optim.SGD(Alpha.parameters(), lr=0.1)

    Alpha.to(device)
    criterion.to(device)

    m = torch.randn(B, N+1, 1, device=device) + 1
    m_valid = torch.randn(B_valid, N+1, 1, device=device) + 1
    expectSig = torch.zeros(signatory.signature_channels(2, depth))
    expectSig[0]=1
    
    mode.initialize(typeVec)
    benchmark = mode.benchmark(bm, w0, initial)
    benchmark_loss = torch.mean(mode.benchmark_loss(initial)).item()
    print("Benchmark loss on train data: ", benchmark_loss)
    mode.initialize(typeVec_test)
    benchmark_test = mode.benchmark(bm_test, w0_test, initial_test)
    Xbar_test = mode.Xbar.clone()
    pi_test = mode.pi

    target_addr = sys.path[0]+'/output'
    # run training
    N_round = 500
    N_epoch = 1
    best_meandiff = float('inf')
    last_linear_coef = np.array([[0]*signatory.signature_channels(2, depth)])
    valid_util = []
    for r in range(N_round):
        optimizer.param_groups[0]['lr'] = .1/(5**(r//200))
        print("round: ", r+1)
        best_valid_loss = float('inf')
        for epoch in range(N_epoch):
            start_time = time.time()
        
            train_loss = utils.train1(Alpha, train_DataLoader,
                                    optimizer, criterion, initial,
                                    m, device)
       
            valid_loss = utils.evaluate1(Alpha, valid_DataLoader, 
                                    criterion, initial_valid,
                                    m_valid, device)
            
            end_time = time.time()
            epoch_min, epoch_sec = utils.epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1} | Epoch Time: {epoch_min}m {epoch_sec:.2f}s')
            print(f'\tTrain Loss: {train_loss:.5f}')
            print(f'\t Val. Loss: {valid_loss:.5f}')
            
        valid_util.append(valid_loss)
        
        Alpha.to('cpu')
        X = Alpha(bm, w0, typeVec, m.to('cpu'), initial)
        #next_m = mode.distFlow(X[:, :, 1:], rough_w0, depth=depth)
        ####################################################################
        mode.distFlow(X[:, :, 1:], rough_w0, depth=depth)
        mode.linear.coef_ = last_linear_coef* r/(r+1) + mode.linear.coef_/(r+1)
        last_linear_coef = mode.linear.coef_
        
        next_m = mode.getDistFlow(B, w0, depth)
        ####################################################################
        
        next_m_valid = mode.getDistFlow(B_valid, w0_valid, depth)
    
        
        m = next_m.detach().to(device)
        m_valid = next_m_valid.detach().to(device)

        e = utils.L2distance(benchmark, X[:, :, 1:].cpu())
        print("The L2 distance between controlled SDE is: ", e)
        Alpha.to(device)
        
    torch.save(Alpha.state_dict(), 'Alpha1.net')
    torch.save(last_linear_coef, target_addr+"/last_linear_coef_Invest.pt")
    with open(target_addr + '/valid_util_Invest.npy', 'wb') as f:
        np.save(f, np.array(valid_util))
    
    
    #utils.plotErrors(errors, target_addr, "L2 Errors", "L2 Errors")
    #utils.plotMeanDiff_bencmarkvspredicted([[i/N for i in range(N+1)], m[:2, :].cpu().detach().numpy(), 
    #                     torch.mean(initial)+w0[:2]* torch.mean(pi*sigma)+ X[:2, :, :1].cpu().detach()*torch.mean(pi*mu)], 
    #                    target_addr, 'Predict vs. Benchmark', 'benchmark')

    ### test
    f = open(target_addr+"/testout", "w")
    sys.stdout = f
    print("-----------------------------")
    print("----------Test Data----------")
    print("-----------------------------")

    Alpha.load_state_dict(torch.load('Alpha1.net'))
    #mode.linear = best_linear
    print("The linear functional: " +str(mode.linear.intercept_)+"," +str(mode.linear.coef_))
    m_test = mode.getDistFlow(B, w0_test, depth)
    utils.plotMeanDiff_bencmarkvspredicted([[i/N for i in range(N+1)], Xbar_test[:3, :].cpu().detach().numpy(), 
                        m_test[:3, :].cpu().detach().numpy()], 
                        target_addr, 
                        r'$\bar{X}$: Predicted vs. Benchmark', 'Xbar_Invest')
    print("L2 loss of Xbar on test data: ", utils.L2distance(Xbar_test, m_test))
    X = Alpha(bm_test.to(device), w0_test.to(device), typeVec_test.to(device), m_test.to(device), initial_test.to(device))
    pi_pred = torch.cat([Alpha.strategy[i].view(-1, 1) for i in range(N)], dim=1)
    testLoss = utils.evaluate1(Alpha, test_DataLoader, criterion, initial_test, m_test.to(device), device)
    print("Loss on test data: ", testLoss)
    print("The L2 distance between controlled SDE on test data: ", utils.L2distance(benchmark_test, X[:, :, 1:].cpu()))
    title = "Controlled SDE on test data"
    name = "Test result"
    utils.plotSDE(benchmark_test[:2].cpu().detach().numpy(), X[:2, :, 1:].cpu().detach().numpy(), 
                  target_addr, title, name)
    
    pi_test = torch.cat([pi_test for i in range(N)], dim=1)
    
    
    print("The L2 distance between pi on test data: ", utils.L2distance(pi_test.view(B, -1, 1), pi_pred.cpu().view(B, -1, 1)))
    title = "Invest on test data"
    name = "Test result invest $$\pi_t$$"
    utils.plotpi(pi_test[:3].cpu().detach().numpy(), pi_pred[:3].cpu().detach().numpy(), 
                  target_addr, title, name)
    f.close()
    
    
if case == "InvestConsumption":
    depth = 4
    rough_w0 = signatory.Path(augment(w0), depth, basepoint=False) # rough path
    rough_w0_valid = signatory.Path(augment(w0_valid), depth, basepoint=False)
    print("Example: InvestConsumption")
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
    
    delta = torch.rand((B, 1))/2+2
    mu = torch.rand((B, 1))/4+0.1
    nu = torch.rand((B, 1))/5 + 0.2
    theta = torch.rand((B, 1))
    sigma = torch.rand((B, 1))/5 +0.2
    eps = torch.rand((B, 1))/2 + 0.5
    typeVec_test = torch.cat([delta, mu, nu, theta, sigma, eps], dim=1)
    
    
    delta = torch.rand((B_valid, 1))/2+2
    mu = torch.rand((B_valid, 1))/4+0.1
    nu = torch.rand((B_valid, 1))/5 + 0.2
    theta = torch.rand((B_valid, 1))
    sigma = torch.rand((B_valid, 1))/5 +0.2
    eps = torch.rand((B_valid, 1))/2 + 0.5
    typeVec_valid = torch.cat([delta, mu, nu, theta, sigma, eps], dim=1)
    
    dat = torch.utils.data.TensorDataset(bm, w0, typeVec) 
    train_DataLoader = torch.utils.data.DataLoader(dat, batch_size=2**11)
    dat = torch.utils.data.TensorDataset(bm_test, w0_test, typeVec_test) 
    test_DataLoader = torch.utils.data.DataLoader(dat, batch_size=2**13)
    dat = torch.utils.data.TensorDataset(bm_valid, w0_valid, typeVec_valid) 
    valid_DataLoader = torch.utils.data.DataLoader(dat, batch_size=2**13)
    
    
    Alpha = model.Action2(train_args, mode)
    criterion = model.LossTotal2(mode, depth)
    optimizer = optim.SGD(Alpha.parameters(), lr=0.5)

    Alpha.to(device)
    criterion.to(device)
    
    m = torch.rand(B, N+1, 1, device=device) + 0.5
    mc = torch.rand(B, N+1, 1, device=device) + 0.5
    m_valid = torch.rand(B_valid, N+1, 1, device=device) + 0.5
    mc_valid = torch.rand(B_valid, N+1, 1, device=device) + 0.5
    
    mode.initialize(typeVec)
    benchmark, benchmark_c = mode.benchmark(bm, w0, initial)
    
    #mc = torch.exp(torch.cat([torch.mean(torch.log(benchmark_c), dim=0).view(1, N, 1) for i in range(B)], dim=0))
    #ss = [torch.cat([mode.pi, benchmark_c[:, i]], dim=1) for i in range(N)]
    #m, gamma = mode.distFlow(mode.Xbar.clone(), mc, rough_w0, depth=depth)
    #m2, _ = mode.distFlow(benchmark, benchmark_c, rough_w0, depth=depth)
    #bench_loss = criterion(torch.cat([benchmark, benchmark], dim=2), mode.Xbar, ss, mc)
    #print("Benchmark loss is : ", bench_loss)
    ## bench_loss= [-2.5977, -2.5864, -2.5912, -2.6119, -2.5882, -2.6010, -2.5942, -2.5922, -2.5987, -2.5897] # 10 runs over 2**15 sample data
    ## theoretical benchmark_loss = (2.5857+2.5899+2.5827)/3=2.5861

#    print("The L2 distance between Xbar on test data: ", utils.L2distance(mode.Xbar.view(B, -1, 1), m.cpu().view(B, -1, 1)))
#    print("The L2 distance between Gamma on test data: ", utils.L2distance(gamma.view(B, -1, 1), mc.cpu().view(B, -1, 1)))
#    print("The L2 distance between Xbar fit and Gamma fit", utils.L2distance(m.view(B, -1, 1), m2.cpu().view(B, -1, 1)))
#    target_addr = sys.path[0]+'/output'
#    utils.plotMeanDiff_bencmarkvspredicted([[i*T/N for i in range(N+1)], mode.Xbar[:2, :].cpu().detach().numpy(),
#                        m[:2, :].cpu().detach().numpy()],
#                        target_addr,
#                        r'$\bar{X}$: Predicted vs. Benchmark (Fit True)', 'Xbar')
#    utils.plotMeanDiff_bencmarkvspredicted([[i*T/N for i in range(N)], mc[:2, :].cpu().detach().numpy(),
#                        gamma[:2, :].cpu().detach().numpy()],
#                        target_addr,
#                        r'$\Gamma$: Predicted vs. Benchmark (Fit True)', 'Gamma')
#    utils.plotMeanDiff_bencmarkvspredicted([[i*T/N for i in range(N+1)], (m.view(B, -1, 1)-m2.view(B, -1, 1))[:2, :].cpu().detach().numpy(),
#                        (m.view(B, -1, 1)-m2.view(B, -1, 1))[2:4, :].cpu().detach().numpy()],
#                        target_addr,
#                        r'$\Gamma$: Xbar vs. Benchmark (Diff)', 'Error')
#    print("Plot done.")
    
    
    
    mode.initialize(typeVec_valid)
    benchmark_valid, benchmark_valid_c = mode.benchmark(bm_valid, w0_valid, initial_valid)
    
    mode.initialize(typeVec_test)
    benchmark_test, benchmark_test_c = mode.benchmark(bm_test, w0_test, initial_test)
    pi_test = mode.pi.clone()
    Xbar_test = mode.Xbar.clone()
    
    N_round = 600
    N_epoch = 1
    valid_util = []
    last_linear_coef = np.array([[0]*signatory.signature_channels(2, depth)])
    last_linearc_coef = np.array([[0]*signatory.signature_channels(2, depth)])
    for r in range(N_round):
        optimizer.param_groups[0]['lr'] = 0.1/(5**(r//200))
        #if r < 100:
        #    optimizer.param_groups[0]['lr'] = 0.5/(5**(r//50)) 
        #else:
        #    optimizer.param_groups[0]['lr'] = 0.5/(10**(r//50))
        print("round: ", r+1)
        best_valid_loss = float('inf')
        for epoch in range(N_epoch):
            start_time = time.time()
        
            train_loss = utils.train2(Alpha, train_DataLoader,
                                    optimizer, criterion, initial,
                                    m, mc, device)
       
            valid_loss = utils.evaluate2(Alpha, valid_DataLoader, 
                                    criterion, initial_valid,
                                    m_valid, mc_valid, device)
            valid_util.append(-valid_loss)
            end_time = time.time()
            epoch_min, epoch_sec = utils.epoch_time(start_time, end_time)

            if r == N_round-1 and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(Alpha.state_dict(), 'Alpha2.net')
        
            print(f'Epoch: {epoch+1} | Epoch Time: {epoch_min}m {epoch_sec:.2f}s')
            print(f'\tTrain Loss: {train_loss:.5f}')
            print(f'\t Val. Loss: {valid_loss:.5f}')
        Alpha.to('cpu')
        X = Alpha(bm, w0, typeVec, m.to('cpu'), mc.to('cpu'), initial)
        c = torch.cat([Alpha.strategy[i][:, 1:] for i in range(N)], dim=1)
        
        mode.distFlow(X[:, :, 1:], c, rough_w0, depth=depth)
        mode.linear.coef_ = r/(r+1)*last_linear_coef + mode.linear.coef_/(r+1)
        mode.linearc.coef_ = r/(r+1)*last_linearc_coef + mode.linearc.coef_/(r+1)
        next_m, next_c = mode.getDistFlow(B, w0, depth)
        last_linear_coef = mode.linear.coef_
        last_linearc_coef = mode.linearc.coef_
        
        
        X_valid = Alpha(bm_valid, w0_valid, typeVec_valid, m_valid.to('cpu'), mc_valid.to('cpu'), initial_valid)
        m_valid, mc_valid = mode.getDistFlow(B_valid, w0_valid, depth)
        m_valid = m_valid.detach().to(device)
        mc_valid = mc_valid.detach().to(device)
        
        m = next_m.detach().to(device)
        mc = next_c.detach().to(device)
        
        print("The L2 distance between controlled SDE is: ", utils.L2distance(benchmark, X[:, :, 1:].cpu()))
        print("The L2 distance between valid controlled SDE is: ", utils.L2distance(benchmark_valid, X_valid[:, :, 1:].cpu()))
        
        Alpha.to(device)
    
    
    torch.save(last_linear_coef, sys.path[0]+"/output/last_linear_coef_InvestConsumption.pt")
    torch.save(last_linearc_coef, sys.path[0]+"/output/last_linearc_coef_InvestConsumption.pt")
    print("The linear functional: " +str(mode.linear.intercept_)+"," +str(mode.linear.coef_))
    with open(sys.path[0]+'/output/valid_util_InvestConsumption.npy', 'wb') as f:
        np.save(f, np.array(valid_util))
    
    ## write to testout.txt file
    import sys
    target_addr = sys.path[0]+'/output'
    
    f = open("testout", "w")
    sys.stdout = f
    ### test
    print("-----------------------------")
    print("----------Test Data----------")
    print("-----------------------------")
    print("The linear functional: " +str(mode.linear.intercept_)+"," +str(mode.linear.coef_))
    print("The linearc functional: " +str(mode.linearc.intercept_)+"," +str(mode.linearc.coef_))
    #Alpha.load_state_dict(torch.load('Alpha2.net'))

    m_test, mc_test = mode.getDistFlow(B, w0_test, depth)
    ##
    testLoss = utils.evaluate2(Alpha, test_DataLoader, criterion, 
                              initial_test, m_test.to(device), 
                              mc_test.to(device), device)
    
    Alpha.to('cpu')
    Xtest = Alpha(bm_test, w0_test, typeVec_test,
              m_test, mc_test, 
              initial_test)
    
    print("Loss on test data: ", testLoss)
    print("The L2 distance between controlled SDE on test data: ", 
          utils.L2distance(benchmark_test, Xtest[:, :, 1:].cpu()))
    print("The L2 (relative) distance between controlled SDE on test data: ", 
          utils.L2distance(benchmark_test, Xtest[:, :, 1:].cpu())/utils.L2distance(benchmark_test, torch.zeros(benchmark_test.size())))
    title = "Controlled SDE on test data"
    name = "Test result"
    utils.plotSDE(benchmark_test[9:11].cpu().detach().numpy(), Xtest[9:11, :, 1:].cpu().detach().numpy(),
                  target_addr, title, name)
    
    ctest = torch.cat([Alpha.strategy[i][:, 1:] for i in range(N)], dim=1)
    ptest = torch.cat([Alpha.strategy[i][:, :1] for i in range(N)], dim=1)
    
    print("The L2 distance between cost c on test data: ", 
          utils.L2distance(benchmark_test_c, ctest.cpu().view(B, -1, 1)))
    print("The L2 (relative) distance between cost c on test data: ", 
          utils.L2distance(benchmark_test_c, ctest.cpu().view(B, -1, 1))/utils.L2distance(benchmark_test_c, torch.zeros(benchmark_test_c.size())))
          #/utils.L2distance(benchmark_test_c, torch.zeros(benchmark_test_c.size()))
    title = "Cost on test data"
    utils.plotC(benchmark_test_c[9:11].cpu().detach().numpy(), ctest[9:11].cpu().detach().numpy(),
                  target_addr, title, name)
    
    benchmark_test_mc = torch.exp(torch.mean(torch.log(benchmark_test_c), dim=0))
    title = r"$Gamma_t$ on test data"
    name = "Test result (dist.)"
    utils.plotmC(benchmark_test_mc.view(N, -1).cpu().detach().numpy(), mc_test[9:11].cpu().detach().numpy(), 
                  target_addr, title, name)
    
    utils.plotMeanDiff_bencmarkvspredicted([[i/N for i in range(N+1)], Xbar_test[9:11, :].cpu().detach().numpy(), 
                        m_test[9:11, :].cpu().detach().numpy()], 
                        target_addr, 
                        r'$\bar{X}$: Predicted vs. Benchmark', 'Xbar')
    print("The L2 distance between Xbar on test data: ", 
          utils.L2distance(Xbar_test.view(B, -1, 1), m_test.cpu().view(B, -1, 1)))
    print("The L2 (relative) distance between Xbar on test data: ", 
          utils.L2distance(Xbar_test.view(B, -1, 1), m_test.cpu().view(B, -1, 1))/utils.L2distance(Xbar_test.view(B, -1, 1), torch.zeros(Xbar_test.size())))
    
    
    pi_test = torch.cat([pi_test for i in range(N)], dim=1)
    
    print("The L2 distance between pi on test data: ", 
          utils.L2distance(pi_test.view(B, -1, 1), ptest.cpu().view(B, -1, 1)))
    print("The L2 (relative) distance between pi on test data: ", 
          utils.L2distance(pi_test.view(B, -1, 1), ptest.cpu().view(B, -1, 1))/utils.L2distance(pi_test.view(B, -1, 1), torch.zeros(B, N, 1)))
    title = "Invest on test data"
    name = "Test result invest $$\pi_t$$"
    utils.plotpi(pi_test[9:11].cpu().detach().numpy(), ptest[9:11].cpu().detach().numpy(),
                  target_addr, title, name)
    f.close()
    
    
    
    
    
