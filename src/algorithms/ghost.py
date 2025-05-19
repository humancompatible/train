from typing import Iterable, Tuple,Callable, List
import pandas as pd
import numpy as np
import scipy as sp
from copy import deepcopy
from scipy.optimize import linprog
from qpsolvers import solve_qp
# import autoray as ar
import timeit

from src.algorithms.Algorithm import Algorithm
from src.algorithms.c_utils.constraint import FairnessConstraint
from src.algorithms.c_utils.constraint_fns import *

from fairret.statistic import *
from src.algorithms.utils import net_grads_to_tensor, net_params_to_tensor
import torch


class StochasticGhost(Algorithm):
    def __init__(self, net, data, loss, constraints):
        super().__init__(net, data, loss, constraints)
       
    @staticmethod
    def solvesubp(fgrad, cval, cgrad, kap_val, beta, tau, hesstype, mc, n, qp_solver='osqp', solver_params={}):
        if hesstype == 'diag':
            P = tau*sp.sparse.identity(n, format='csc')
            kap = kap_val * np.ones(mc)
            cval = np.array(cval)
        return solve_qp(
            P,
            fgrad.reshape((n,)),
            cgrad.reshape((mc, n)),
            kap-cval,
            np.zeros((0, n)),
            np.zeros((0,)),
            -beta*np.ones((n,)),
            beta*np.ones((n,)),
            qp_solver)


    @staticmethod
    def compute_kappa(cval, cgrad, lamb, rho,mc ,n):
        term1 = (1 - lamb) * np.maximum(cval, 0).max()
        obj = np.zeros(n + 1)
        obj[0] = 1.0
        A_ub = np.hstack([-np.ones((mc, 1)), cgrad])
        b_ub = -cval
        bounds = [(0, None)] + [(-rho, rho) for _ in range(n)]

        try:
            res = linprog(c=obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                term2 = lamb * res.fun
            else:
                term2 = lamb * rho
        except:
            term2 = lamb * rho

        return term1 + term2
     
       
    def optimize(self,
                 geomp, stepsize_rule = 'inv_iter', zeta=0.5, gamma0 = 0.1, rho=0.8, lamb=0.5, beta=10., tau=1.,
                 device='cpu', seed = None, verbose = True,
                 max_runtime = None, max_iter = None):
        
        
        max_sample_size = np.max([c.group_sizes() for c in self.constraints])
        n = sum(p.numel() for p in self.net.parameters())
        
        rng = np.random.default_rng(seed=seed)
        run_start = timeit.default_timer()
        
        for iteration in range(0, max_iter):
        
            current_time = timeit.default_timer()
            self.history['time'].append(current_time - run_start)
            
            if max_runtime > 0 and current_time - run_start >= max_runtime:
                print(current_time - run_start)
                self.history['constr'] = pd.DataFrame(self.history['constr'])
                return self.history
        
            if stepsize_rule == 'inv_iter':
                gamma = gamma0/(iteration+1)**zeta
            elif stepsize_rule == 'dimin':
                if iteration == 0:
                    gamma = gamma0
                else:
                    gamma *= (1-zeta*gamma)
            
            Nsamp = rng.geometric(p=geomp) - 1
            while (2**(Nsamp+1)) > max_sample_size:
                Nsamp = rng.geometric(p=geomp) - 1
        
            self.history['n_samples'].append(3*(1 + 2 ** (Nsamp+1)))
            dsols = np.zeros((4, n))
            
            ################
            ### sampling ###
            ################
            indices_f = []
            samples_c = []
            
            subp_batch_size = 2**(Nsamp+1)
            
            indices_f.append(rng.choice(len(self.dataset), size=1))
            samples_c.append([c.sample_dataset(1) for c in self.constraints])
            
            idx_f = rng.choice(len(self.dataset), size=subp_batch_size)
            indices_f.extend([idx_f[::2], idx_f[1::2], idx_f])
            s_c = [c.sample_dataset(subp_batch_size) for c in self.constraints]
            samples_c.extend([
                [[(x[::2], y[::2]) for x, y in c_sample] for c_sample in s_c],
                [[(x[1::2], y[1::2]) for x, y in c_sample] for c_sample in s_c],
                s_c
            ])
            
            ##############
            ### update ###
            ##############
            for j, samples in enumerate(zip(indices_f, samples_c)):
                self.net.zero_grad()
                
                idx = samples[0]
                obj_batch = self.dataset[idx]
                c_batch = samples[1]

                
                # calculate autograd jacobian of obj fun w.r.t. params
                outs = self.net(obj_batch[0])
                if obj_batch[1].ndim < outs.ndim:
                    outs = outs.squeeze(1)
                feval = self.loss_fn(outs, obj_batch[1])
        
                dfdw = torch.autograd.grad(feval, self.net.parameters())
                dfdw = torch.concat([dfdwi.flatten() for dfdwi in dfdw])
                
                # calculate autograd jacobian of self.constraints fun w.r.t. params
                
                constraint_eval = []
                dcdw = []
                for i, c in enumerate(self.constraints):
                    self.net.zero_grad()
                    # print(j, i)
                    c_val = c.eval(self.net, c_batch[i])
                    
                    c_grad = torch.autograd.grad(c_val, self.net.parameters())
                    c_grad = torch.concat([cg.flatten() for cg in c_grad]).detach().numpy()
                    
                    constraint_eval.append(c_val.detach())
                    dcdw.append(c_grad)
                    
                constraint_eval = np.array(constraint_eval)
                dcdw = np.array(dcdw)
                
                kappa = self.compute_kappa(constraint_eval, dcdw, rho, lamb, mc=2, n=len(dfdw))
                
                # solve subproblem
                feval = feval.detach().numpy()
                dfdw = dfdw.detach().numpy()
                dsol = self.solvesubp(dfdw,
                                    constraint_eval, dcdw,
                                    kappa, beta, tau,
                                    hesstype='diag', mc=2, n=len(dfdw),
                                    qp_solver='osqp')
                
                dsols[j, :] = dsol
            
            # aggregate solutions to the subproblem according to Eq. 23
            dsol = dsols[0, :] + (dsols[3, :]-0.5*dsols[1, :] -
                                0.5*dsols[2, :])/(geomp*((1-geomp)**(Nsamp)))
            
            start = 0
            print(f'{iteration}', end='\r')
            with torch.no_grad():
                w = net_params_to_tensor(self.net)
                if any([torch.any(torch.isnan(lw)) for lw in w]):
                    print('NaNs!')
                    return self.history
                for i in range(len(w)):
                    end = start + w[i].numel()
                    w[i].add_(torch.tensor(gamma*np.reshape(dsol[start:end], np.shape(w[i]))))
                    start = end
                    
            self.history['w'].append(deepcopy(self.net.state_dict()))
            
            feval = self.loss_fn(outs, obj_batch[1])
        
        self.history['constr'] = pd.DataFrame(self.history['constr'])
        return self.history