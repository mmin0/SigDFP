

import torch
from sklearn.linear_model import Ridge
import signatory

class Invest():
    def __init__(self, params):
        
        self.T = params["T"]
        self.N = params["N"]
        self.device = params["device"]
        
        
    def initialize(self, typeVec):
        """
        input:
            size -- tuple (batch, 1)
        initialize random parameters
        """
        self.delta = typeVec[:, 0:1]
        self.mu = typeVec[:, 1:2]
        self.nu = typeVec[:, 2:3]
        self.theta = typeVec[:, 3:4]
        self.sigma = typeVec[:, 4:5]
        
    
    def running(self, x, m, alpha):
        """
        input:
            alpha -- torch.tensor([pi, c])
        """
        return 0
    
    def terminal(self, x, mx):
        
        return torch.exp(-1/self.delta*(x - self.theta*mx))
        
    def one_step_simulation(self, x, m, alpha, dw, dcn):
        """
        input:
            x.size() = x.size(m)
        """
        dt = self.T/self.N
        x_next = x + alpha*(dt*self.mu + dw*self.nu + dcn*self.sigma)
        return x_next
    
    
    def benchmark(self, w, cn, initial):
        """
        input:
            m -- tensor(batch, N+1, dim), distribution interaction
            w -- tensor(batch, N, dim), brownian increments
            cn -- tensor(batch, N+1, dim), common noise
            initial -- tensor(batch, 1, dim), starting point, has initial distribution mu_0
        return:
            X -- tensor(batch, N+1, dim), benchmark paths, no extra time dimension
            loss -- benchmark loss calculated by Monte Carlo from X
        """
        phi = torch.mean(self.delta*self.mu*self.sigma/(self.sigma**2+self.nu**2))
        psi = torch.mean(self.theta*self.sigma**2/(self.sigma**2+self.nu**2))
        
        
        pi = self.delta*self.mu/(self.sigma**2+self.nu**2) +\
                self.theta*self.sigma/(self.sigma**2+self.nu**2) * phi/(1-psi)
        batch, _, dim = w.size()
        
        X = torch.zeros(batch, self.N+1, dim)
        X[:, 0, :] = initial
        for i in range(1, self.N+1):
            X[:, i, :] = self.one_step_simulation(X[:, i-1, :], 0, 
                                                 pi, w[:, i-1, :], 
                                                 cn[:, i]-cn[:, i-1])
        self.pi = pi
        self.Xbar = torch.zeros(batch, self.N+1, dim)
        self.Xbar[:, 0] = torch.mean(initial)
        dt = self.T/self.N
        e1 = torch.mean(pi*self.mu)
        e2 = torch.mean(pi*self.sigma)
        for i in range(1, self.N+1):
            self.Xbar[:, i] = self.Xbar[:, i-1] + e1*dt + e2*(cn[:, i]-cn[:, i-1])
        return X

        
    def distFlow(self, X, rough, in_dim=2, depth=2):
        """
        input:
            rough -- signatory.Path, rough path object of common noise
            
        return:
            mx, mc -- next round conditional dist.
        """
        batch, _, _ = X.size()
        m = torch.zeros(batch, self.N+1, 1)
        
        m[:, 0] = torch.mean(X[:, 0])
        
        
        self.linear = Ridge(alpha=.1, tol=1e-6)
        data = torch.cat([rough.signature().cpu().detach(),
                          rough.signature(None, self.N//2+1),
                          torch.zeros(batch, signatory.signature_channels(in_dim, depth))],
                        dim=0)
        label = torch.cat([X[:, -1].cpu().detach(), X[:, self.N//2].cpu().detach(), X[:, 0].cpu().detach()], dim=0)
        self.linear.fit(data.numpy(), label.numpy())
        l = torch.tensor(self.linear.coef_).view(-1, 1)
        
        for i in range(2, self.N+2):
            m[:, i-1] = torch.matmul(rough.signature(end=i), l) + self.linear.intercept_
        return m
        

    def getDistFlow(self, batch, w0, depth):
        augment = signatory.Augment(1, 
                                    layer_sizes = (), 
                                    kernel_size = 1,
                                    include_time = True)
        
        rough = signatory.Path(augment(w0), depth)
        m = torch.zeros(batch, self.N+1, 1)
        m[:, 0] = torch.tensor(self.linear.intercept_)
        
        l = torch.tensor(self.linear.coef_, dtype=torch.float).view(-1, 1)
        # i=1
        
        for i in range(2, self.N+2):
            m[:, i-1] = torch.matmul(rough.signature(end=i), l) + self.linear.intercept_
        
        return m
    
    
    
    def benchmark_loss(self, initial):
        phi = torch.mean(self.delta*self.mu*self.sigma/(self.sigma**2+self.nu**2))
        psi = torch.mean(self.theta*self.sigma**2/(self.sigma**2+self.nu**2))
        
        pi = self.delta*self.mu/(self.sigma**2+self.nu**2) +\
                self.theta*self.sigma/(self.sigma**2+self.nu**2) * phi/(1-psi)
        rho = (self.mu + self.theta/self.delta*self.sigma* torch.mean(self.sigma*pi))**2/2/(self.sigma**2+self.nu**2)\
                -self.theta/self.delta*torch.mean(self.mu*pi)-0.5*(self.theta/self.delta*torch.mean(self.sigma*pi))**2
        return torch.mean(torch.exp(-(initial-self.theta*torch.mean(initial))/self.delta)*torch.exp(-rho * self.T))
    
    