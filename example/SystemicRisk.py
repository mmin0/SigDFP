
import numpy as np
import torch
from sklearn.linear_model import Ridge
import signatory

class SystemicRisk():
    def __init__(self, params):
        self.sigma = params["sigma"]
        self.q = params["q"]
        self.a = params["a"]
        self.eps = params["eps"]
        self.rho= params["rho"]
        self.N = params["N"]
        self.T = params["T"]
        self.c = params["c"]
        self.device = params["device"]
        R = self.a**2 + 2*self.a*self.q + self.eps
        self.delta_p = -(self.a+self.q) + np.sqrt(R)
        self.delta_m = -(self.a+self.q) - np.sqrt(R)
        
    def initialize(self, size, device):
        return
    
    def running(self, x, m, alpha):
        """
        input:
            x.size() = x.size(m)
        """
        return 0.5*torch.pow(alpha, 2) - self.q*alpha*(m-x) + 0.5*self.eps*torch.pow(m-x,2)
    
    def terminal(self, x, m):
        """
        input:
            x.size() = x.size(m)
        """
        return 0.5*self.c*torch.pow(m-x, 2)
        
    def one_step_simulation(self, x, m, alpha, dw, dcn):
        """
        input:
            x.size() = x.size(m)
        """
        dt = self.T/self.N
        return x + (self.a*(m-x)+alpha)*dt + self.sigma*(self.rho*dcn + np.sqrt(1-self.rho**2)*dw)
    
    def eta(self, t):
        numerator = -(self.eps-self.q**2)*(np.exp((self.delta_p-self.delta_m)*(self.T-t))-1)\
                    -self.c*(self.delta_p*np.exp((self.delta_p - self.delta_m)*(self.T-t)) - self.delta_m)
        denominator = self.delta_m*np.exp((self.delta_p-self.delta_m)*(self.T-t))-self.delta_p \
                        -self.c*(np.exp((self.delta_p-self.delta_m)*(self.T-t)) -1)
        return numerator/denominator
    
    def benchmark(self, w, cn, initial):
        """
        input:
            m -- tensor(batch, N+1, dim), distribution interaction
            w -- tensor(batch, N, dim), brownian increments
            cn -- tensor(batch, N+1, dim), common noise
            initial -- tensor(batch, 1, dim), starting point, has initial distribution mu_0
        return:
            X -- tensor(batch, N+1, dim), benchmark paths, no extra time dimension
        """
        dt = self.T/self.N
        batch, _, dim = w.size()
        
        X = torch.zeros(batch, self.N+1, dim)
        X[:, 0, :] = initial
        self.alpha = torch.zeros(batch, self.N, 1)
        for i in range(1, self.N+1):
            m = torch.mean(initial) + self.rho*self.sigma*cn[:, i]
            alpha = (self.q + self.eta(dt*i-dt))*(m-X[:, i-1])
            X[:, i, :] = self.one_step_simulation(X[:, i-1, :], m, 
                                                 alpha, w[:, i-1, :], 
                                                 cn[:, i]-cn[:, i-1])
            self.alpha[:, i-1] = alpha
        return X
    
    def benchmark_loss(self, initial):
        
        mu = 0
        dt = self.T/self.N
        for i in range(self.N):
            mu += 0.5*self.sigma**2*(1-self.rho**2) * self.eta(self.T/self.N*i) * dt
        return torch.mean(mu + 0.5*self.eta(0)*(torch.mean(initial)-initial)**2).item()
    
        
    def distFlow(self, X, rough, in_dim=2, depth=1):
        """
        input:
            rough -- signatory.Path, rough path object of common noise
            
        return:
            m -- next round conditional dist.
        """
        batch, _, _ = X.size()
        m = torch.zeros(batch, self.N+1, 1)
        
        m[:, 0] = torch.mean(X[:, 0])
        self.linear = Ridge(alpha=.1, tol=1e-6)
        data = torch.cat([rough.signature().cpu().detach(),
                          rough.signature(end=self.N//2+1).cpu().detach(),
                          torch.zeros(batch, signatory.signature_channels(in_dim, depth))],
                        dim=0)
        label = torch.cat([X[:, -1].cpu().detach(), X[:, self.N//2].cpu().detach(), X[:, 0].cpu().detach()], dim=0)
        self.linear.fit(data.numpy(), label.numpy())
        l = torch.tensor(self.linear.coef_).view(-1, 1)
        # i=1
        
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
