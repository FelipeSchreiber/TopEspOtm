import bisect
import numpy as np

M = 1e12
## Assume que recebe um problema na forma 
## Min c^Tx
## s.t.
## Ax=b
class LP:
    def __init__(self,A,b,c)-> None:
        idx = np.where(b < 0)
        self.A = A
        self.b = b
        self.c = c
        self.A[idx] *= -1
        self.b[idx] *= -1

    ## Encontra uma base para o problema
    ## Min c^Tx
    ## s.t.
    ## Ax = b
    def phase_one(self,A,b):
        AI = np.hstack([A,np.eye(A.shape[0])])
        c = np.zeros(AI.shape[1])
        c[-A.shape[0]:] = 1
        # caso for implementar o bigM
        c_M = np.zeros(AI.shape[1])
        c_M[-A.shape[0]:] = M
        c_M[:self.c.shape[0]] = -self.c
        c = c_M
        #print(self.c.shape[0],c_M.shape[0])
        base_idx = list(range(A.shape[1],AI.shape[1]))
        assert len(base_idx) == A.shape[0]
        assert base_idx[-1] == A.shape[0] + A.shape[1] - 1
        base_idx, x, opt,_ = self.simplex(base_idx,AI,b,c)
        if opt < 0:
            print("Não há solução") #somente no caso das variaveis artificiais
        return base_idx

     ## Simplex
     ##  Min c^Tx
     ##  s.t.
     ##  Ax = B
    def simplex(self,base_idx,A,b,c):
        vals = []
        while True:
            mask = np.zeros(A.shape[1],dtype=bool)
            mask[base_idx] = True
            non_base_idx = np.where(~mask)[0]
            B = A[:,mask]
            N = A[:,~mask]
            B_inv = np.linalg.inv(B)
            BN = B_inv@N
            cb = c[mask]
            z = cb@BN
            xb = B_inv@b
            x = np.zeros(mask.shape[0])
            x[base_idx] = xb
            diff = z - c[~mask]
            #enter_ind = np.argmax(diff)
            if (diff <= 0).all():
                return base_idx, x, c@x,vals
            ## primeiro indice que melhora a funcao
            enter_ind = np.where(diff > 0)[0][0] 
            y = BN[:,enter_ind]
            if (y <= 0).all():
                x = np.zeros(mask.shape[0])
                x[base_idx] = xb
                print("Solucao ilimitada")
                return base_idx, x, c@x, vals
            leave_ind = 0
            for i in range(len(y)):
                if y[i] > 0:
                    leave_ind = i
                    break
            for i in range(len(y)):
                if y[i] > 0:
                    if xb[i]/y[i] < xb[leave_ind]/y[leave_ind]:
                        leave_ind = i
            del base_idx[leave_ind]
            bisect.insort(base_idx,non_base_idx[enter_ind])
            vals.append(c@x)
    ##Dual simplex
    ## Max c^Tx
    ## s.t.
    ## Ax = b
    def dual_simplex(self,base_idx,A,b,c):
        while True:
            mask = np.zeros(A.shape[1],dtype=bool)
            mask[base_idx] = True
            non_base_idx = np.where(~mask)[0]
            B = A[:,mask]
            N = A[:,~mask]
            B_inv = np.linalg.inv(B)
            BN = B_inv@N
            cb = c[mask]
            u = cb@B_inv
            xb = B_inv@b
            leave_ind = np.argmin(xb)
            z = ((u@N) - c[~mask])
            for i in range(len(z)):
                if BN[leave_ind,i] != 0:
                    z[i] /= BN[leave_ind,i]
                else:
                    z[i] = np.nan
            # enter_ind = np.argmax(z)
            enter_ind = np.where(z < 0)[0][0] 
            if (xb >= 0).all():
                x = np.zeros(mask.shape[0])
                x[base_idx] = xb
                return base_idx, x, c@x 
            del base_idx[leave_ind]
            bisect.insort(base_idx,non_base_idx[enter_ind])
        
    def solve(self):
        base_idx = self.phase_one(self.A,self.b)
        print("Encontrou uma base")
        # if base_idx > self.A.shape[1]:
        #     print("inviável")
        #     return None
        base_opt,x,opt_val,vals = self.simplex(base_idx,self.A,self.b,self.c)
        return x,opt_val,vals