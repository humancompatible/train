from ot.utils import list_to_array
from ot.backend import get_backend
import warnings
import argparse
import numpy as np
import time
from scipy.optimize import linprog
from qpsolvers import solve_qp
from ot.utils import unif, dist, list_to_array
import autoray as ar
# from .backend import get_backend


def makeparms(maxiter=1, beta=10, rho=0.8, lamb=0.5, hess='diag', tau=1., mbsz=1, numcon=1, geomp=0.7, stepdecay='dimin', gammazero=0.1, zeta=0.1):
    params = {
        'maxiter': maxiter,  # number of iterations performed
        'beta': beta,  # trust region size
        'rho': rho,  # trust region for feasibility subproblem
        'lamb': lamb,  # weight on the subfeasibility relaxation
        'hess': hess,  # method of computing the Hessian of the QP, options include 'diag' 'lbfgs' 'fisher' 'adamdiag' 'adagraddiag'
        'tau': tau,  # parameter for the hessian
        'mbsz': mbsz,  # the standard minibatch size, used for evaluating the progress of the objective and constraint
        'numcon': numcon,  # number of constraint functions
        'geomp': geomp,  # parameter for the geometric random variable defining the number of subproblem samples
        'stepdecay': stepdecay, # strategy for step decrease, options include 'dimin' 'stepwise' 'slowdimin' 'constant'
        'gammazero': gammazero,  # initial stepsize
        'zeta': zeta,  # parameter associated with the stepsize iteration
    }
    return params


# def computekappa( cval, cgrad, rho, lamb, mc, n):
#     obj = np.concatenate(([1.], np.zeros((n,))))
#     Aubt = np.concatenate((([-1.]), cgrad))
#     # if there are multiple constraints? Aubt.reshape(mc,n+1) ??
#     Aubt = Aubt.reshape(mc, n+1)
#     res = linprog(c=obj, A_ub=Aubt, b_ub=[-cval], bounds=(-rho, rho))
#     return ((1-lamb)*max(0, cval)+lamb*max(0, res.fun))

def computekappa(cval, cgrad, lamb, rho, mc, n, scalef):  
    obj = np.concatenate(([1.], np.zeros((n,))))
    Aubt = np.column_stack((-np.ones(mc), np.array(cgrad)))
    try:
       res = linprog(c=obj, A_ub=Aubt, b_ub=-np.array(cval), bounds=[(-rho, rho)])
       #print("IMPORTANT!!!!!",res.fun)
       return (1-lamb)*max(0, sum(cval)) + lamb*max(0, res.fun)
    except:
       return (1-lamb)*max(0, sum(cval)) + lamb*max(0, rho)
    #res = linprog(c=obj, A_ub=Aubt, b_ub=-np.array(cval), bounds=[(-rho, rho)])
    #return (1-lamb)*max(0, sum(cval)) + lamb*max(0, res.fun)
    #return lamb*max(0, res.fun)


# def solvesubp(fgrad, cval, cgrad, kap, beta, tau, hesstype, mc, n):
#     if hesstype == 'diag':
#        #P = tau*nx.eye(n)
#        P = tau*np.identity(n)
#     return solve_qp(P, fgrad.reshape((n,)), cgrad.reshape((mc, n)), list_to_array([(kap-cval)]), np.zeros((0, n)), np.zeros((0,)), -beta*np.ones((n,)), beta*np.ones((n,)), solver='osqp')


def solvesubp(fgrad, cval, cgrad, kap_val, beta, tau, hesstype, mc, n):
    if hesstype == 'diag':
       # P = tau*nx.eye(n)
       P = tau*np.identity(n)
       kap = kap_val * np.ones(mc)
       cval = np.array(cval)
    return solve_qp(P, fgrad.reshape((n,)), cgrad.reshape((mc, n)), kap-cval, np.zeros((0, n)), np.zeros((0,)), -beta*np.ones((n,)), beta*np.ones((n,)), solver='osqp')

# initw : Initial parameters of the Network (Weights and Biases)


def StochasticGhost(obj_fun, obj_grad, con_funs, con_grads, initw, params):
    N = params["N"]
    n = params["n"]  
    maxiter = params["maxiter"]
    beta = params["beta"]
    rho = params["rho"]
    lamb = params["lamb"]
    tau = params["tau"]
    hess = params["hess"]
    mbsz = params["mbsz"]
    mc = params["numcon"]
    geomp = params["geomp"]
    stepdec = params["stepdecay"]
    gamma0 = params["gammazero"]
    zeta = params["zeta"]
    gamma = gamma0
    lossbound = params["lossbound"]
    scalef = params["scalef"]
    
    w = initw
    for i in range(len(w)):
        w[i] = ar.to_numpy(w[i])

    feval = obj_fun(w, mbsz)  
    ceval = np.zeros((mc,))
    #ceval_white = np.zeros((mc,))
    #ceval_black = np.zeros((mc,))
    Jeval = np.zeros((mc, n))

    # Getting all the constraints
    iterfs = np.zeros((maxiter,))
    iterfs[0] = feval
    dir_obj = np.zeros((maxiter,))
    dir_cons = np.zeros((maxiter, mc))
    obj_grad_norm = np.zeros((maxiter,))
    cons_grad_norm = np.zeros((maxiter, mc))
    for i in range(mc):
       conf = con_funs[i]
       #ceval[i] = np.max(conf(w, mbsz), 0)
       ceval[i] = np.max(conf(w, mbsz), 0)
    #itercs = np.zeros((maxiter,))
    itercs = np.zeros((maxiter, mc))
    #itercs_black = np.zeros((maxiter, mc))
    #itercs_white = np.zeros((maxiter, mc))
    itercs[0,:] = np.max(ceval)

    for iteration in range(0, maxiter):

        if stepdec == 'dimin':
           gamma = gamma0/(iteration+1)**zeta
        if stepdec == 'constant':
           gamma = gamma0
        if stepdec == 'slowdimin':
           gamma = gamma*(1-zeta*gamma)
        if stepdec == 'stepwise':
           gamma = gamma0 / (10**(int(iteration*zeta)))

        Nsamp = np.random.geometric(p=geomp)
        while (2**(Nsamp+1)) > N:
          Nsamp = np.random.geometric(p=geomp)

        # Only specify the number of minibatches here
        # Lets the user decide on the samples for each minibatch number
        mbatches = [1, 2**Nsamp, 2**Nsamp, 2**(Nsamp+1)]
        dsols = np.zeros((4, n))

        for j in range(4):
          feval = obj_fun(w, mbatches[j])
          fgrad = ar.to_numpy(obj_grad(w, mbatches[j]))
          #print(type(fgrad))
          # cval = []
          # cgrad = []
          for i in range(mc):
            # con_funs[i](conf) and con_grads[i](conJ) ith constraint and constraint grad
            conf = con_funs[i]
            conJ = con_grads[i]
            # ceval and Jeval are evaluations of ith constraint and constraint grads for the parameter values
            # nx.max(conf(w,mbatches[j]),0) to ensure the problem is always in the feasible region
            #ceval[i] = np.max(conf(w, mbatches[j]) - lossbound[i], 0)
            ceval[i] = np.max(conf(w, mbatches[j]) - lossbound[i], 0)
            Jeval[i, :] = ar.to_numpy(conJ(w, mbatches[j]))
            
            #print(type(Jeval[i, :]))
            # cval = nx.concatenate((cval, ceval[i]))
            # cgrad = nx.concatenate((cgrad, Jeval[i, :]))

          # Compute Kappa for the Subproblem bound   
          kap = computekappa(ceval, Jeval, rho, lamb, mc, n, scalef)
          # Solving the subproblem
          dsol = solvesubp(fgrad, ceval, Jeval, kap, beta, tau, hess, mc, n)
          dsols[j, :] = dsol

        dsol = dsols[0, :] + (dsols[3, :]-0.5*dsols[1, :] -
                              0.5*dsols[2, :])/(geomp*((1-geomp)**Nsamp))
         
        print(">>>>>step_norm<<<<<<",np.linalg.norm(dsol))
        #print("ITERATION", iteration)
        #print("step vector is: ")
        # w = w + gamma*dsol
        # The stepsize evaluation from the previously calculated gradients
        start = 0
        for i in range(len(w)):
           #print(w[i].size)
           end = start + np.size(w[i])
           #print("Parameter ", i+1, np.reshape(dsol[start:end], np.shape(w[i])))
           w[i] = w[i] + gamma*np.reshape(dsol[start:end], np.shape(w[i]))
           start = end
        
        feval = obj_fun(w, mbsz)
        # dir der purposes
        fgrad = obj_grad(w, mbsz)
        # dir der purposes
        iterfs[iteration] = feval
        for i in range(mc):
          conf = con_funs[i]
          ceval[i] = np.max(conf(w, mbsz), 0)
          Jeval[i, :] = ar.to_numpy(conJ(w, mbsz))
          #ceval[i] = np.max(conf(w, mbsz)[0], 0)
          #ceval_black[i] = np.max(conf(w, mbsz)[1], 0)
          #ceval_white[i] = np.max(conf(w, mbsz)[2], 0)
        itercs[iteration, :] = ceval
        #itercs_black[iteration, :] = ceval_black
        #itercs_white[iteration, :] = ceval_white
        obj_grad_norm[iteration] = np.linalg.norm(fgrad)
        cons_grad_norm[iteration, :] = np.array([np.linalg.norm(Jeval[i]) for i in range(mc)])
        dir_obj[iteration] = np.dot(fgrad, dsol)
        dir_cons[iteration, :] = np.array([np.dot(Jeval[i], dsol) for i in range(mc)])

    return w, iterfs, itercs
