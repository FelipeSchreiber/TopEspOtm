import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian, grad
from equality_constrained import newton_equality_constrained
from unconstrained import grad_irrestricted

c = 1e-3
def make_stop_cond(fs,max_):##True para max, False para sum
    k=0
    comparison = None
    if max_:
        k = 1
        comparison = np.max
    else:
        k = len(fs) - 1
        comparison = np.sum
    def stop_cond(x,g,t):
        a = np.array([f(x[k:]) for f in fs[1:]])
        #n_a = np.where(a <= 0, 0, a)
        b = np.array([f((x-t*g)[k:]) for f in fs[1:]])
        #n_b = np.where(b <= 0, 0, b)
        r = comparison(b) - comparison(a)
        return r > 1e-5 and (a > 0).any()
    return stop_cond

def make_obj_func(fs,t):
    def f(x):
        return t*fs[0](x) - np.sum([np.log(-f(x)) for f in fs[1:]])
    return f

def make_obj_func_phase_one_max(fs,t):
    def f_max(x):
        return t*x[0] - np.sum([np.log(x[0] - f(x[1:])) for f in fs[1:]],axis=0)
    return f_max

def make_obj_func_phase_one_sum(fs,t,m):
    #as primeiras m variaveis do vetor x são os S's, 
    # cada uma correspondendo a uma das m restricoes 
    def f_sum(x):
        return t*np.sum(x[:m]) - np.sum([np.log(x[i] - f(x[m:])) + np.log(x[i]) 
                                        for i,f in enumerate(fs[1:]) ],axis=0)
    return f_sum

class interior_points:
    def __init__(self,fs,A,b,x_0,accuracy=1e-3,mu=10,method_max = True):
        self.A = A
        self.b = b
        self.fs = fs
        self.x_ini = x_0
        self.acc = accuracy
        self.mu = mu ##fator que multiplica o t no metodo barreira
        self.t_0 = 1
        self.m = len(fs) - 1 ##quantidade de restricoes. f_0 é funcao objetivo,
        # logo é desconsiderada
        self.method_max = method_max ##seleciona o metodo de fase 1, 
        #inviabilidade maxima ou soma das inviabilidades

    def phase_one_max(self):
        ##begins barrier method
        ##inicializa s como sendo o valor máximo dentre todas as funcoes de desigualdade
        s = np.amax([f_(self.x_ini) for f_ in self.fs[1:]]) + c
        ##insere s no começo do vetor de variaveis X 
        x = np.insert(self.x_ini,0,s,axis=0)
        ##inicializa t com t_0
        t = self.t_0
        stop_cond = make_stop_cond(self.fs,True)
        while self.m/t > self.acc: 
            f = make_obj_func_phase_one_max(self.fs,t)
            g = grad(f)
            solver = grad_irrestricted(f,g,x_0=x,stop_cond=stop_cond)
            x,_,_ = solver.solve()
            t *= self.mu
        print("Max infeasibility: ",x[0])
        return x[1:]

    def phase_one_sum(self):
        ##begins barrier method
        ##inicializa s como sendo o valor máximo dentre todas as funcoes de desigualdade
        s = np.array([f_(self.x_ini) + c for f_ in self.fs[1:]])
        ##insere s no começo do vetor de variaveis X 
        x = np.insert(self.x_ini,0,s,axis=0)
        ##inicializa t com t_0
        t = self.t_0
        stop_cond = make_stop_cond(self.fs,False)
        while self.m/t > self.acc:
            ## obtem as funcoes objetivo, gradiente e hessiana considerando a barreira log e 
            # em funcao de t 
            f = make_obj_func_phase_one_sum(self.fs,t,self.m)
            g = grad(f)
            solver = grad_irrestricted(f,g,x_0=x,stop_cond=stop_cond)
            x,_,_ = solver.solve()
            t *= self.mu
        print("Sum infeasibilities: ",np.sum(x[:self.m]))
        return x[self.m:]

    def phase_two(self,x_0):
        ##begins barrier method
        ##inicializa t com t_0
        x=x_0.copy()
        t = self.t_0
        vals = []
        total_nt_steps = []
        lambda_sqrs = []
        while self.m/t > self.acc:
            ## obtem as funcoes objetivo, gradiente e hessiana considerando a barreira log e 
            # em funcao de t
            f = make_obj_func(self.fs,t)
            g = grad(f)
            h = jacobian(egrad(f))
            solver = newton_equality_constrained(f,g,h,
                                                self.A.copy(),
                                                self.b.copy(),
                                                x.copy()
                                               )
            x, nu,i,_,_,lambda_sqr = solver.solve()
            lambda_sqrs.extend(lambda_sqr)
            vals.append(self.m/t)
            t *= self.mu
            total_nt_steps.append(i)
        return x,nu,vals,total_nt_steps,lambda_sqrs

    def solve(self):
        x = 0
        nu = 0
        n = 0
        ##executa fase um
        if self.method_max:
            ##chama o metodo de fase 1 com maxima inviabilidade
            x = self.phase_one_max()
        else:
            ##chama o metodo de fase 1 com a soma das inviabilidades
            x = self.phase_one_sum()
        print("Begins phase two")
        x,nu,vals,nt_steps,lambdas_sqr = self.phase_two(x)
        ## checa se a solucao é boa o suficiente
        return x,nu,vals,nt_steps,lambdas_sqr