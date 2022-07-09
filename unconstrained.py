# import numpy as np
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian, grad
import math

from numpy import NaN

def stop_condition_gen(eta):
    def stop_condition(x,grad,t): 
        return np.linalg.norm(grad) > eta
    return stop_condition

def line_search_gen(f,alpha,beta,grad):
    def line_search(v,x):
        t = 1
        #include restrictions i.e. the domain
        g = grad(x)
        l = NaN
        while math.isnan(l):
            l = f(x+t*v)
            t *= beta
        while f(x+t*v) > f(x) + alpha*t*(g.dot(v)):
            t *= beta
        return t
    return line_search

def gen_nt_step(grad,hess):
    def newton_step(x):
        h = hess(x)
        g = grad(x)
        h_inv = np.linalg.inv(h)
        step = h_inv@g
        lambda_ = g.dot(step)
        return -step, lambda_
    return newton_step 

class newton_irrestricted:
    def __init__(self,f,g,h,x_0,eta=1e-5,alpha=0.01,beta=0.5):
        self.f = f
        self.grad = grad(f)
        self.hess = jacobian(egrad(f))
        self.x_ini = x_0
        self.nt_step = gen_nt_step(self.grad,self.hess)
        self.stop_condition = stop_condition_gen(eta)
        self.line_search = line_search_gen(self.f,alpha,beta,self.grad)
    
    def solve(self):
        x = self.x_ini
        v, l = self.nt_step(x)
        i = 0
        sols = []
        t = 0
        while self.stop_condition(l,-v,t=t):
            v, l = self.nt_step(x)
            t = self.line_search(v,x)
            x += t*v
            i +=1
            sols.append(self.f(x))
        return x,i,sols

class grad_irrestricted:
    def __init__(self,f,g,x_0,eta=1e-5,alpha=0.01,beta=0.5,autodiff=False,stop_cond=None):
        self.f = f
        if autodiff:
            self.grad = grad(f)
        else:
            self.grad = g
        self.x_ini = x_0
        if stop_cond == None:
            self.stop_condition = stop_condition_gen(eta)
        else:
            self.stop_condition = stop_cond
        self.line_search = line_search_gen(self.f,alpha,beta,self.grad) 
    
    def solve(self):
        x = self.x_ini
        g = self.grad(x)
        i = 0
        t = 1
        sols = []
        while True:
            g = self.grad(x)
            v = -1*g
            t = self.line_search(v,x)
            if not self.stop_condition(x,g,t):
                return x,i,sols
            x += t*v
            i += 1
            sols.append(self.f(x))