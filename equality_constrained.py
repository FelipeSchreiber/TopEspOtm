from autograd import elementwise_grad as egrad
from autograd import jacobian, grad
import autograd.numpy as np
import math

def gen_residual(grad,A,b):
    def residual(x,nu):
        return np.hstack([grad(x) + (A.T)@nu, A@x - b])
    return residual

def stop_condition_gen(residual,A,b,eta):
    def stop_condition(x,nu):
        c1 = np.linalg.norm(residual(x,nu)) > eta
        # c2 = (A@x != b).any()
        return c1
    return stop_condition

def line_search_gen(f,residual,alpha,beta):
    def line_search(x,delta_x,nu,delta_nu):
        t = 1
        l = math.nan
        while math.isnan(l):
            l = f(x+t*delta_x)
            t *= beta
        while np.linalg.norm(residual(x + t*delta_x, nu+t*delta_nu)) > \
            (1 - alpha*t)*np.linalg.norm(residual(x, nu)):
            t *= beta
        return t
    return line_search

def gen_nt_step(grad,hess,A,b):
    def newton_step(x,nu):
        KKT = np.vstack([
                    np.hstack([hess(x), A.T]),
                    np.hstack([A, np.zeros((A.shape[0],A.shape[0]))])
                        ])
        h = A.dot(x) - b
        B = -1*np.hstack([grad(x) + A.T@nu, h])
        delta = np.linalg.solve(KKT,B)
        return delta[:x.shape[0]], delta[x.shape[0]:]
    return newton_step 

class newton_equality_constrained:
    def __init__(self,f,g,h,A,b,x_0,eta=1e-3,alpha=0.1,beta=0.5,autodiff=False):
        self.f = f
        self.grad = g
        self.hess = h
        if autodiff:
            self.grad = grad(f)
            self.hess = jacobian(egrad(f))
        self.residual = gen_residual(self.grad,A,b)
        self.x_ini = x_0
        self.m = A.shape[0]
        self.nt_step = gen_nt_step(self.grad,self.hess,A,b)
        self.stop_condition = stop_condition_gen(self.residual,A,b,eta)
        self.line_search = line_search_gen(self.f,self.residual,alpha,beta)
    
    def solve(self):
        x = self.x_ini
        i = 0
        nu= np.random.rand(self.m)
        t = 1
        r_pri = []
        r_dual = []
        lambda_sqr = []
        while self.stop_condition(x,nu):
            delta_x, delta_nu = self.nt_step(x,nu)
            t = self.line_search(x,delta_x,nu,delta_nu)
            x += t*delta_x
            nu += t*delta_nu
            i+=1
            r = self.residual(x,nu)
            r_pri.append(np.linalg.norm(r[x.shape[0]:]))
            r_dual.append(np.linalg.norm(r[:x.shape[0]]))
            lambda_sqr.append((-self.grad(x)@delta_x)/2)
        return x,nu,i,r_pri,r_dual,lambda_sqr