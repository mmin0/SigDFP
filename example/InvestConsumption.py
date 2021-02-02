
import torch
from sklearn.linear_model import Ridge
import signatory

class InvestConsumption():
    def __init__(self, params):
        
        self.T = params["T"]
        self.N = params["N"]
        self.device = params["device"]
        
        
    def initialize(self, typeVec):
        """
        input:
            typeVec
        initialize random parameters
        """
        
        self.delta = typeVec[:, 0:1]
        self.mu = typeVec[:, 1:2]
        self.nu = typeVec[:, 2:3]
        self.theta = typeVec[:, 3:4]
        self.sigma = typeVec[:, 4:5]
        self.eps = typeVec[:, 5:6]
            
        
        
        
    def U(self, x):
        """
        input:
            x -- torch.tensor, positive
            delta -- torch.tensor
        """
        return 1/(1-1/self.delta)*torch.pow(x, 1-1/self.delta)
        
    
    def running(self, x, mx, alpha, mc):
        """
        input:
            alpha -- torch.tensor([pi, c])
        """
        
        c = alpha[:, 1:]
        return -self.U(c*x*torch.pow(mc*mx, -self.theta))
        

    
    def terminal(self, x, mx):
        
        return -self.eps*self.U(x*mx**(-self.theta))
        
    def one_step_simulation(self, x, pi, c, dw, dcn):
        """
        input:
            x.size() = x.size(m)
        """
        dt = self.T/self.N
        x_next = torch.log(x) + pi*(dt*self.mu + dw*self.nu + dcn*self.sigma) - c*dt\
                    - 0.5* pi**2 *(self.sigma**2 + self.nu**2)*dt
        
        return torch.exp(x_next)
    
    def c(self, t):
        '''
        if self.beta == 0:
            return 1/(self.T - t + 1/self.lam)
        else:
            return 1/(1/self.beta + (1/self.lam - 1/self.beta)*torch.exp(-self.beta*(self.T-t)))
        '''
        return 1/(1/self.beta + (1/self.lam - 1/self.beta)*torch.exp(-self.beta*(self.T-t)))
        
    def benchmark(self, w, cn, initial):
        """
        input:
            w -- tensor(batch, N, dim), brownian increments
            cn -- tensor(batch, N+1, dim), common noise
            initial -- tensor(batch, 1, dim), starting point, has initial distribution mu_0
        return:
            X -- tensor(batch, N+1, dim), benchmark paths, no extra time dimension
            c -- tensor(batch, N, dim), benchmark cost
        """
        phi = torch.mean(self.delta*self.mu*self.sigma/(self.sigma**2+self.nu**2))
        psi = torch.mean(self.theta*(self.delta-1)*self.sigma**2/(self.sigma**2+self.nu**2))
        q = phi/(1+psi)
        e1 = torch.mean((self.delta*self.mu**2-self.theta*(self.delta-1)*self.sigma*self.mu*q)/(self.sigma**2+self.nu**2))
        e2 = torch.mean((self.delta*self.mu-self.theta*(self.delta-1)*self.sigma*q)**2/(self.sigma**2+self.nu**2))
        rho = (1-1/self.delta)*(\
                self.delta/(2*(self.sigma**2+self.nu**2))*(self.mu-self.sigma*q*self.theta*(1-1/self.delta))**2 \
              + 0.5*q**2*self.theta**2*(1-1/self.delta) - self.theta*e1 + 0.5*self.theta*e2)
        self.beta = (self.theta*(self.delta-1)*torch.mean(self.delta*rho)/(1+torch.mean(self.theta*(self.delta-1))) - self.delta*rho)
        q1 = self.eps**(-self.delta)
        self.lam =  q1 * torch.exp(torch.mean(torch.log(q1)))**(-self.theta*(self.delta-1)/(1+torch.mean(self.theta*(self.delta-1)))) 
        
        pi = self.delta*self.mu/(self.sigma**2+self.nu**2) - self.theta*(self.delta-1)*self.sigma/(self.sigma**2+self.nu**2)*q
        dt = self.T/self.N
        batch, _, dim = w.size()
        
        X = torch.zeros(batch, self.N+1, dim)
        c = torch.zeros(batch, self.N, dim)
        X[:, 0, :] = initial
        for i in range(1, self.N+1):
            c[:, i-1, :] = self.c(i*dt-dt)
            X[:, i, :] = self.one_step_simulation(X[:, i-1, :], 
                                                 pi.view(-1, 1), self.c(i*dt-dt), w[:, i-1, :], 
                                                 cn[:, i]-cn[:, i-1])
        self.pi = pi
        
        self.Xbar = torch.zeros(batch, self.N+1, dim)
        self.Xbar[:, 0, :] = torch.mean(torch.log(initial))
        d1 = torch.mean(pi*self.mu - 0.5*pi**2 *(self.sigma**2 + self.nu**2))*dt
        d2 = torch.mean(self.sigma*pi)
        for i in range(1, self.N+1):
            self.Xbar[:, i, :] = self.Xbar[:, i-1, :] + d1 - torch.mean(c[:, i-1, :])*dt + d2*(cn[:, i]-cn[:, i-1])
        self.Xbar = torch.exp(self.Xbar)
        #############
        
        f = 0
        for i in range(self.N):
            f += (rho+1/self.delta*c[:, i, :]+torch.mean(c[:, i, :])*(1-1/self.delta)*self.theta)*dt
        f = torch.exp(f)
        self.benchmark_loss = torch.mean(self.eps/(1-1/self.delta)*initial**(1-1/self.delta)*self.Xbar[:, 0, :]**(-self.theta*(1-1/self.delta))*f)
        
        return X, c
    
    def benchmark_loss(self):
        return self.benchmark_loss
        
    def distFlow(self, X, c, rough, in_dim=2, depth=4):
        """
        input:
            rough -- signatory.Path, rough path object of common noise
            
        return:
            mx, mc -- next round conditional dist.
        """
        
        batch, _, _ = X.size()
        mx = torch.zeros(batch, self.N+1, 1)
        mc = torch.zeros(batch, self.N, 1)
        
        #mx[:, 0] = torch.exp(torch.mean(torch.log(X[:, 0])))
        
        self.linear = Ridge(alpha=0.1, tol=1e-6)
        
        data = torch.cat([rough.signature(None, None).cpu().detach(),
                          rough.signature(None, self.N//2+1).cpu().detach(),
                          torch.zeros(batch, signatory.signature_channels(in_dim, depth))],
                        dim=0)
        label = []
        label.append(X[:, -1].cpu().detach())
        label.append(X[:, self.N//2].detach())
        label.append(X[:, 0].cpu().detach())
        
        label = torch.log(torch.cat(label, dim=0))
        
        self.linear.fit(data.numpy(), label.numpy())
        
        l = torch.tensor(self.linear.coef_, dtype=torch.float32).view(-1, 1)
        # i=1
        mx[:, 0] = torch.exp(torch.tensor([self.linear.intercept_]))
        for i in range(2, self.N+2):
            mx[:, i-1] = torch.exp(torch.matmul(rough.signature(end=i), l) + self.linear.intercept_)
            
        self.linearc = Ridge(alpha=0.1, tol=1e-5)
        labelc = torch.log(torch.cat([c[:, -1].cpu().detach(), 
                                      c[:, self.N//2].detach(),
                                      c[:, 0].cpu().detach()], dim=0))
        
        
        self.linearc.fit(data.numpy(), labelc.numpy())
        
        lc = torch.tensor(self.linearc.coef_).view(-1, 1)
        mc[:, 0] = torch.exp(torch.tensor([self.linearc.intercept_]))
        for i in range(2, self.N+1):
            mc[:, i-1] = torch.exp(torch.matmul(rough.signature(end=i), lc) + self.linearc.intercept_)
        return mx, mc
        
        
      

    def getDistFlow(self, batch, w0, depth):
        augment = signatory.Augment(1, 
                                    layer_sizes = (), 
                                    kernel_size = 1,
                                    include_time = True)
        
        rough = signatory.Path(augment(w0), depth)
        mx = torch.zeros(batch, self.N+1, 1)
        mx[:, 0] = torch.exp(torch.tensor(self.linear.intercept_))
        mc = torch.zeros(batch, self.N, 1)
        mc[:, 0] = torch.exp(torch.tensor(self.linearc.intercept_))
        
        l = torch.tensor(self.linear.coef_, dtype=torch.float32).view(-1, 1)
        lc = torch.tensor(self.linearc.coef_, dtype=torch.float32).view(-1, 1)
        # i=1
        
        for i in range(2, self.N+2):
            mx[:, i-1] = torch.exp(torch.matmul(rough.signature(end=i), l) + self.linear.intercept_)
            
        for i in range(2, self.N+1):
            mc[:, i-1] = torch.exp(torch.matmul(rough.signature(end=i), lc) + self.linearc.intercept_)
        return mx, mc
