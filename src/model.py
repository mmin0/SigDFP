

import torch
import torch.nn as nn


class Action(nn.Module):
    def __init__(self, args, mode):
        """
        input:
            args.in_dim -- the dimension of (t, X_t)
            args.neurons -- the size of hidden layers
            args.out_dim -- the dimension of alpha
        """
        super(Action, self).__init__()
        indim = args.in_dim + 1
        self.linear = nn.ModuleList([nn.Linear(indim, args.neurons[0])])
        for i in range(len(args.neurons)-1):
            self.linear.append(nn.Linear(args.neurons[i], args.neurons[i+1]))
        self.linear.append(nn.Linear(args.neurons[-1], args.out_dim))
        self.mode = mode
        
        
        
    def forward(self, bm, cn, m, initial):
        """
        input:
            bm -- tensor(batch, N, dim), brownian increments
            cn -- tensor(batch, N+1, dim), common noise
            m -- tensor(batch, N+1, dim), mu from previous step
            initial -- starting point
        return:
            tensor(batch, N+1, dim), generate paths of controlled SDE
        """
        device = bm.device
        self.strategy = []
        batch, N, _ = bm.size()
        X = torch.zeros(batch, N+1, 2, device=device)
        X[:, 0, 1:] = initial #torch.randn(batch, 1, device=device)
        
        for i in range(1, N+1):
            X[:, i, 0] = i/N
            self.strategy.append(self.one_step(X[:, i-1, :].clone(), m[:, i]))
            
            X[:, i, 1:] = self.mode.one_step_simulation(X[:, i-1, 1:], m[:, i],
                                                         self.strategy[-1], bm[:, i-1],
                                                         cn[:, i]-cn[:, i-1])
        return X
        
    
    def one_step(self, x, mt):
        """
        input:
            x -- the augmented data (t, X_t)
            mt -- conditional distribution
        return:
            alpha -- torch.tensor(batch, dim), control
        """
        x = torch.cat([x, mt], dim=1)
        alpha = torch.relu(self.linear[0](x))
        for i in range(1, len(self.linear)-1):
            alpha = torch.relu(self.linear[i](alpha))
        alpha = self.linear[-1](alpha)
        return alpha



## search for nonconstant EQ
class Action1(nn.Module):
    def __init__(self, args, mode):
        """
        input:
            args.in_dim -- the dimension of typeVec
            args.neurons -- the size of hidden layers
            args.out_dim -- the dimension of alpha
        """
        super(Action1, self).__init__()
        indim = args.in_dim + 3 # type vector dimensions +(t,x,m)
        self.linear = nn.ModuleList([nn.Linear(indim, args.neurons[0])])
        self.bn = nn.BatchNorm1d(args.in_dim)
        for i in range(len(args.neurons)-1):
            self.linear.append(nn.Linear(args.neurons[i], args.neurons[i+1]))
        self.linear.append(nn.Linear(args.neurons[-1], args.out_dim))
        self.mode = mode
        
        
        
    def forward(self, bm, cn, typeVec, m, initial):
        """
        input:
            typeVec -- type vector
            bm -- tensor(batch, N, dim), brownian increments
            cn -- tensor(batch, N+1, dim), common noise
            m -- tensor(batch, N+1, dim), mu from previous step
            initial -- starting point
        return:
            tensor(batch, N+1, dim), generate paths of controlled SDE
        """
        device = bm.device
        self.strategy = []
        batch, N, _ = bm.size()
        X = torch.zeros(batch, N+1, 2, device=device)
        X[:, 0, 1:] = initial #torch.randn(batch, 1, device=device)
        
        self.mode.initialize(typeVec)
        
        
        for i in range(1, N+1):
            X[:, i, 0] = i/N*self.mode.T
            self.strategy.append(self.one_step(typeVec, X[:, i-1].clone(), m[:, i-1]))
            X[:, i, 1:] = self.mode.one_step_simulation(X[:, i-1, 1:], m[:, i],
                                                         self.strategy[-1], bm[:, i-1],
                                                         cn[:, i]-cn[:, i-1])
        return X
        
    
    def one_step(self, typeVec, x, m):
        """
        input:
            typeVec -- type vector
        return:
            alpha -- torch.tensor(batch, dim), control
        """
        x = torch.cat([self.bn(typeVec), x, m], dim=1)
        alpha = torch.relu(self.linear[0](x))
        for i in range(1, len(self.linear)-1):
            alpha = torch.relu(self.linear[i](alpha))
        alpha = self.linear[-1](alpha)
        return alpha
    


class Action2(nn.Module):
    def __init__(self, args, mode):
        """
        used for invest consumption model, since this model produce strategy (pi, c)
        input:
            args.in_dim -- the dimension of typeVec
            args.neurons -- the size of hidden layers
            args.out_dim -- the dimension of alpha
        """
        super(Action2, self).__init__()
        indim = args.in_dim + 4
        self.bn = nn.BatchNorm1d(args.in_dim)
        self.bnc = nn.BatchNorm1d(args.in_dim)
        self.linear = nn.ModuleList([nn.Linear(indim, args.neurons[0])])
        for i in range(len(args.neurons)-1):
            self.linear.append(nn.Linear(args.neurons[i], args.neurons[i+1]))
        self.linear.append(nn.Linear(args.neurons[-1], 1))
        
        self.linearc = nn.ModuleList([nn.Linear(indim, args.neurons[0])])
        for i in range(len(args.neurons)-1):
            self.linearc.append(nn.Linear(args.neurons[i], args.neurons[i+1]))
        self.linearc.append(nn.Linear(args.neurons[-1], 1))
        self.mode = mode
        
        
        
    def forward(self, bm, cn, typeVec, mx, mc, initial):
        """
        input:
            bm -- tensor(batch, N, dim), brownian increments
            cn -- tensor(batch, N+1, dim), common noise
            mx -- tensor(batch, N+1, dim), from previous step
            mc -- tensor(batch, N, dim), from previous step
            initial -- starting point
        return:
            tensor(batch, N+1, dim), generate paths of controlled SDE
        """
        device = bm.device
        self.strategy = []
        batch, N, _ = bm.size()
        X = torch.zeros(batch, N+1, 2, device=device)
        X[:, 0, 1:] = initial #torch.randn(batch, 1, device=device)
        self.mode.initialize(typeVec)
        
        for i in range(1, N+1):
            X[:, i, 0] = i/N*self.mode.T
            pi = self.one_step_pi(typeVec, X[:, i-1, :].clone(), mx[:, i-1], mc[:, i-1])
            c = self.one_step_c(typeVec, X[:, i-1, :].clone(), mx[:, i-1], mc[:, i-1])
            self.strategy.append(torch.cat([pi, c], 
                                            dim=1))
            
            X[:, i, 1:] = torch.relu(self.mode.one_step_simulation(X[:, i-1, 1:].clone(),
                                                         pi, c, bm[:, i-1],
                                                         cn[:, i]-cn[:, i-1])-0.0001)+0.0001
        return X
        
    
    def one_step_pi(self, typeVec, x, mt, ct):
        """
        input:
            x -- the augmented data (t, X_t)
            mt -- conditional averaged state
            ct -- conditional averaged consumption
        return:
            alpha -- torch.tensor(batch, dim), control
        """
        pi = torch.cat([self.bn(typeVec), x, mt, ct], dim=1)
        #pi = (pi-torch.mean(pi, dim=0))/torch.std(pi, dim=0)
        
        for i in range(len(self.linear)-1):
            pi = torch.relu(self.linear[i](pi))
        pi = self.linear[-1](pi)
        return pi
    
    def one_step_c(self, typeVec, x, mt, ct):
        c = torch.cat([self.bnc(typeVec), x, mt, ct], dim=1)
        
        for i in range(len(self.linearc)-1):
            c = torch.relu(self.linearc[i](c))
        c = self.linearc[-1](c)
        
        return torch.exp(c) #torch.relu(c-0.00001)+0.00001



class LossTotal(nn.Module):
    def __init__(self, mode, depth, dim=1):
        """
        input:
            mode -- which example we are running
        """
        super(LossTotal, self).__init__()
        self.mode = mode
        self.dim = dim
        self.depth = depth
        
    
    def forward(self, X, m, strategy):
        """
        input:
            X -- augmented path
            m -- torch.tensor(batch, N+1, dim), the distribution interaction 
                    process from last round simulation, for example \bar{m}_t in 
                    the case of SystemicRisk.
            strategy -- list[N]
        """
        N = len(strategy)
        # control lost
        loss_c = self.mode.terminal(X[:, -1, 1:], m[:, -1])
        for i in range(N):
            loss_c = loss_c + self.mode.running(X[:, i, 1:], m[:, i], strategy[i])/N*self.mode.T
        return torch.mean(loss_c)
    
    
    
    
    
    
class LossTotal2(nn.Module):
    def __init__(self, mode, depth, dim=1):
        """
        input:
            mode -- which example we are running
        """
        super(LossTotal2, self).__init__()
        self.mode = mode
        self.dim = dim
        self.depth = depth
        
    
    def forward(self, X, m, strategy, mc):
        """
        input:
            X -- augmented path
            m -- torch.tensor(batch, N+1, dim), the distribution interaction 
                    process from last round simulation, for example \bar{m}_t in 
                    the case of SystemicRisk.
            strategy -- list[N]
        """
        N = len(strategy)
        # control lost
        loss_c = self.mode.terminal(X[:, -1, 1:], m[:, -1])
        
        for i in range(N):
            #c = strategy[i][:, 1:]
            loss_c = loss_c + self.mode.running(X[:, i, 1:], m[:, i], strategy[i], mc[:, i])/N*self.mode.T
                
        return torch.mean(loss_c)
    
