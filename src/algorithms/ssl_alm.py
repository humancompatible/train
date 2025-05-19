from copy import deepcopy
import timeit
from typing import Callable
import numpy as np
import torch

from src.algorithms.utils import net_grads_to_tensor, net_params_to_tensor, _set_weights
from src.algorithms.Algorithm import Algorithm



class SSLPD(Algorithm):
    def __init__(self, net, data, loss, constraints, custom_project_fn: Callable = None):
        super().__init__(net, data, loss, constraints)
        self.project = custom_project_fn if custom_project_fn else self.project_fn
        
    @staticmethod
    def project_fn(x, m):
        for i in range(1,m+1):
            if x[-i] < 0:
                x[-i] = 0
        return x
       
    def optimize(self,  lambda_bound, eta, rho, tau, mu, beta,
                 batch_size, epochs, start_lambda=None, max_runtime = None, max_iter = None,
                 seed = None, device = 'cpu',
                 verbose = True):
        
        m = len(self.constraints)
        slack_vars = torch.zeros(m, requires_grad=True)
        _lambda = torch.zeros(m, requires_grad=True) if start_lambda is None else start_lambda
        
        z = torch.concat([net_params_to_tensor(self.net, flatten=True, copy=True), slack_vars])
        
        c = self.constraints
                
        run_start = timeit.default_timer()

        for epoch in range(epochs):
            
            gen = torch.Generator(device=device)
            if not (seed is None):
                gen.manual_seed(seed+epoch)
                
            loss_loader = torch.utils.data.DataLoader(self.dataset, batch_size, shuffle=True, generator=gen)
            
            for iteration, (f_inputs, f_labels) in enumerate(loss_loader):
                
                if max_iter is not None and iteration == max_iter:
                    break
                current_time = timeit.default_timer()
                self.history['w'].append(deepcopy(self.net.state_dict()))
                self.history['time'].append(current_time - run_start)
                self.history['n_samples'].append(batch_size*3)
                if max_runtime > 0 and current_time - run_start >= max_runtime:
                    break
                
                ########################
                ## UPDATE MULTIPLIERS ##
                ########################
                self.net.zero_grad()
                slack_vars.grad = None
                
                # sample for and calculate self.constraints (lines 2, 3)
                c_sample = [ci.sample_loader() for ci in c]
                c_1 = [ci.eval(self.net, c_sample[i]).reshape(1) + slack_vars[i] for i, ci in enumerate(c)]
                # update multipliers (line 3)
                with torch.no_grad():
                    _lambda = _lambda + eta * torch.concat(c_1)
                # dual safeguard (lines 4,5)
                if torch.norm(_lambda) >= lambda_bound:
                    _lambda = torch.zeros_like(_lambda, requires_grad=True)
                    
                    
                    
                #######################
                ## UPDATE PARAMETERS ##
                #######################
                outputs = self.net(f_inputs)
                if f_labels.dim() < outputs.dim():
                    f_labels = f_labels.unsqueeze(1)
                loss_eval = self.loss_fn(outputs, f_labels)
                f_grad = torch.autograd.grad(loss_eval, self.net.parameters())
                f_grad = torch.concat([*[g.flatten() for g in f_grad], torch.zeros(m)]) # add zeros for slack vars
                self.net.zero_grad()
                
                # constraint grad estimate
                c_grad = []
                for ci in c_1:
                    ci_grad = torch.autograd.grad(ci, self.net.parameters())
                    slack_grad = torch.autograd.grad(ci, slack_vars)
                    c_grad.append(torch.concat([*[g.flatten() for g in ci_grad], *slack_grad]))
                    self.net.zero_grad()
                    slack_vars.grad = None
                c_grad = torch.stack(c_grad)
                
                # independent constraint estimate
                with torch.no_grad():
                    c_sample = [ci.sample_loader() for ci in c]
                    c_2 = torch.concat([ci.eval(self.net, c_sample[i]).reshape(1) + slack_vars[i] for i, ci in enumerate(c)])
                
                x_t = torch.concat([
                    net_params_to_tensor(self.net, flatten=True, copy=True),
                    slack_vars
                ])
                
                G = f_grad + c_grad.T @ _lambda + rho*(c_grad.T @ c_2) + mu*(x_t - z)
                x_t1 = self.project(x_t - tau*G, m)
                z += beta*(x_t-z)
                with torch.no_grad():
                    _set_weights(self.net, x_t1)
                    for i in range(len(slack_vars)):
                        slack_vars[i] = x_t1[i-len(slack_vars)]
                
                if verbose:
                    with np.printoptions(precision=6, suppress=True, floatmode='fixed'):
                        print(f"""{epoch:2}|{iteration:5} | {loss_eval.detach().cpu().numpy()}|{_lambda.detach().cpu().numpy()}|{c_2.detach().cpu().numpy()}|{slack_vars.detach().cpu().numpy()}""", end='\r')
                        
        ######################
        ### POSTPROCESSING ###    
        ######################
        
        G_hat = torch.zeros_like(G)
        
        f_inputs, f_labels = self.dataset[:][0], self.dataset[:][1]
        cgrad_sample = [ci.sample_dataset(np.inf) for ci in c]
        c_sample = [ci.sample_dataset(np.inf) for ci in c]

        self.net.zero_grad()
        slack_vars.grad = None
        # loss
        outputs = self.net(f_inputs)       
        if f_labels.dim() < outputs.dim():
            f_labels = f_labels.unsqueeze(1)
        loss_eval = self.loss_fn(outputs, f_labels)
        # loss grad
        loss_eval = self.loss_fn(outputs, f_labels)
        f_grad = torch.autograd.grad(loss_eval, self.net.parameters())
        f_grad = torch.concat([*[g.flatten() for g in f_grad], torch.zeros(m)])# add zeros for slack vars
        self.net.zero_grad()
        # constraint grad estimate
        c_1 = [ci.eval(self.net, c_sample[i]).reshape(1) + slack_vars[i] for i, ci in enumerate(c)]
        c_grad = []
        for ci in c_1:
            ci_grad = torch.autograd.grad(ci, self.net.parameters())
            slack_grad = torch.autograd.grad(ci, slack_vars)
            c_grad.append(torch.concat([*[g.flatten() for g in ci_grad], *slack_grad]))
            self.net.zero_grad()
        c_grad = torch.stack(c_grad)
        
        # independent constraint estimate
        with torch.no_grad():
            c_2 = torch.concat([
                ci.eval(self.net, cgrad_sample[i]).reshape(1) + slack_vars[i] for i, ci in enumerate(c)
            ])
        x_t = torch.concat([
            net_params_to_tensor(self.net, flatten=True, copy=True),
            slack_vars
        ])
        G_hat += f_grad + c_grad.T @ _lambda + rho*(c_grad.T @ c_2) + mu*(x_t - z)
            
        x_t1 = self.project(x_t - tau*G_hat, m)
        with torch.no_grad():
            _set_weights(self.net, x_t1)
        
        current_time = timeit.default_timer()
        self.history['w'].append(deepcopy(self.net.state_dict()))
        self.history['time'].append(current_time - run_start)
        self.history['n_samples'].append(batch_size*3)
        
        return self.history