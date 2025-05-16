import argparse
import os
import pandas as pd
from utils.load_folktables import load_folktables_torch
import numpy as np
import torch
from torch import tensor, nn
from torch.utils.data import TensorDataset, Subset
from fairret.statistic import *
from fairret.metric import *
from fairret.loss import NormLoss, LSELoss, KLProjectionLoss
from copy import deepcopy
import timeit

from src.algorithms.c_utils.constraint_fns import one_sided_loss_constr
from src.algorithms.c_utils.constraint import FairnessConstraint
from src.algorithms.ssl_alm import SSLPD
from src.algorithms.switching_subgradient import SSSG
from src.algorithms.ghost import StochasticGhost

class SimpleNet(nn.Module):
    def __init__(self, in_shape, out_shape, dtype):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_shape, 64, dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 32, dtype=dtype),
            nn.ReLU(),
            nn.Linear(32, out_shape, dtype=dtype),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='folktables exp')
    
    ### experiment parameters
    parser.add_argument('-alg', '--algorithm')
    parser.add_argument('-time', '--time', type=int)
    parser.add_argument('-ne', '--num_exp', type=int)
    parser.add_argument('-task', '--task', type=str)
    parser.add_argument('-state', '--state', type=str)
    parser.add_argument('-loss_bound', '--loss_bound', type=float)
    parser.add_argument('-constraint', '--constraint', type=str)
    parser.add_argument('-alg_name', '--alg_name', type=str, nargs='?', const='', default='')
    parser.add_argument('-device', '--device', type=str)
    parser.add_argument('--download', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--data_path', nargs='?', type=str, default=None,const=None)
    
    ### algorithm parameters
    parser.add_argument('-maxiter', '--maxiter', nargs='?', const=None, type=int)
    # ghost
    parser.add_argument('-alpha', '--geomp', nargs='?', const=0.2, default=0.2, type=float)
    parser.add_argument('-beta', '--beta', nargs='?', const=10, default=10, type=float)
    parser.add_argument('-rho', '--rho', nargs='?', const=0.8, default=0.8, type=float)
    parser.add_argument('-lambda', '--lam', nargs='?', const=0.5, default=0.5, type=float)
    parser.add_argument('-gamma0', '--gamma0', nargs='?', const=0.05, default=0.05, type=float)
    parser.add_argument('-zeta', '--zeta', nargs='?', const=0.3, default=0.3, type=float)
    parser.add_argument('-tau', '--tau', nargs='?', const=1, default=1, type=float)
    parser.add_argument('-stepsize', '-sr', nargs='?', const='inv_iter', default='inv_iter', type=str)
    
    # alm
    parser.add_argument('-bs', '--batch_size', nargs='?', const=16, default=16, type=int)
    
    # ssg
    parser.add_argument('-frule', '--frule', nargs='?', const='dimin', default='dimin', type=str)
    parser.add_argument('-fs', '--f_stepsize', nargs='?', const=7e-1, default=7e-1, type=float)
    parser.add_argument('-crule', '--crule', nargs='?', const='dimin', default='dimin', type=str)
    parser.add_argument('-cs', '--c_stepsize', nargs='?', const=7e-1, default=7e-1, type=float)
    parser.add_argument('-epochs', '--epochs', nargs='?', const=4, default=4, type=int)
    parser.add_argument('-ctol', '--ctol', nargs='?', type=float)
    parser.add_argument('-k0', '--save_iter', nargs='?', const=None, default=None, type=int)
    
    # sslalm
    parser.add_argument('-mu', '--mu', nargs='?', const=2, default=2, type=float)
    parser.add_argument('-eta', '--eta', nargs='?', const=1e-2, default=1e-2, type=float)
    
    #fairret
    parser.add_argument('-fconstr', '--fconstr', nargs=1, type=str)
    parser.add_argument('-losstype', '--losstype', nargs=1, type=str)
    parser.add_argument('-mult', '--mult', nargs=1, type=float)

    # parse args
    args = parser.parse_args()
    ALG_TYPE = args.algorithm
    EXP_NUM = args.num_exp
    FT_STATE = args.state
    LOSS_BOUND = args.loss_bound
    CONSTRAINT = args.constraint
    TASK = args.task
    ALG_CUSTOM_NAME = args.alg_name
    MAX_TIME = args.time
    DOWNLOAD_DATA = args.download
    DATA_PATH = args.data_path
    
    if ALG_TYPE.startswith('sgd'):
        epochs = args.epochs
        BATCH_SIZE = args.batch_size
        params_str = f'bs{BATCH_SIZE}'
    elif ALG_TYPE.startswith('fairret'):
        epochs = args.epochs
        BATCH_SIZE = args.batch_size
        fconstr = args.fconstr[0]
        losstype = args.losstype[0]
        mult = args.mult[0]
        params_str = f'bs{BATCH_SIZE}c{fconstr}l{losstype}m{mult}'
    elif ALG_TYPE.startswith('sg'):
        G_ALPHA = args.geomp
        MAXITER_GHOST = 1000 if args.maxiter is None else args.maxiter
        ghost_rho = args.rho
        ghost_beta = args.beta
        ghost_lambda = args.lam
        ghost_gamma0 = args.gamma0
        ghost_zeta = args.zeta
        ghost_tau = args.tau
        ghost_stepsize_rule = args.stepsize
        params_str = f'a{G_ALPHA}rho{ghost_rho}beta{ghost_beta}lambda{ghost_lambda}gamma{ghost_gamma0}zeta{ghost_zeta}tau{ghost_tau}ss{ghost_stepsize_rule}'
    elif ALG_TYPE.startswith('swsg'):
        epochs=args.epochs
        ctol = args.ctol
        BATCH_SIZE = args.batch_size
        f_stepsize_rule=args.frule
        f_stepsize=args.f_stepsize
        c_stepsize_rule=args.crule
        c_stepsize=args.c_stepsize
        save_iter = args.save_iter
        params_str = f'ctol{ctol}fsr{f_stepsize_rule}fs{f_stepsize}csr{c_stepsize_rule}cs{c_stepsize}'
    elif ALG_TYPE.startswith('aug'):
        epochs=args.epochs
        BATCH_SIZE = args.batch_size
        MAXITER_ALM = 1000 if args.maxiter is None else args.maxiter
    elif ALG_TYPE.startswith('sslalm'):
        epochs = args.epochs
        BATCH_SIZE = args.batch_size
        lambda_bound = 100
        rho = args.rho
        mu = args.mu
        tau = args.tau
        beta = args.beta
        eta = args.eta
        params_str = f'mu{mu}rho{rho}tau{tau}eta{eta}beta{beta}'
    
    if ALG_CUSTOM_NAME == '':
        ALG_CUSTOM_NAME = params_str
    ALG_TYPE += '_'+ALG_CUSTOM_NAME
    
    if args.device == 'cpu':
        device = 'cpu'
    elif ALG_TYPE.startswith('sg'):
        device = 'cpu'
        print('CUDA not supported for Stochastic Ghost')
    elif torch.cuda.is_available():
        device = 'cuda'
        print('CUDA found')
    else:
        device = 'cpu'
        print('CUDA not found')
    
    print(f'{device = }')    
    torch.set_default_device(device)
    
    DTYPE = torch.float32

    FT_DATASET = TASK
    torch.set_default_dtype(DTYPE)
    DATASET_NAME = FT_DATASET + '_' + FT_STATE
    
    X_train, y_train, [w_idx_train, nw_idx_train], X_test, y_test, [w_idx_test, nw_idx_test] = load_folktables_torch(
        FT_DATASET, state=FT_STATE.upper(), random_state=42, make_unbalanced = False, onehot=False, download=DOWNLOAD_DATA, path=DATA_PATH
    )
    
    pos_idx_train = np.argwhere(y_train == 1).flatten()
    neg_idx_train = np.argwhere(y_train == 0).flatten()
    pos_idx_test = np.argwhere(y_test == 1).flatten()
    neg_idx_test = np.argwhere(y_test == 0).flatten()
    
    w_idx_train_pos = np.array(set(w_idx_train) & set(pos_idx_train))
    w_idx_train_neg = np.array(set(w_idx_train) & set(neg_idx_train))
    nw_idx_train_pos = np.array(set(nw_idx_train) & set(pos_idx_train))
    nw_idx_train_neg = np.array(set(nw_idx_train) & set(neg_idx_train))
        
    X_train_tensor = tensor(X_train, dtype=DTYPE)
    y_train_tensor = tensor(y_train, dtype=DTYPE)
    train_ds = TensorDataset(X_train_tensor,y_train_tensor)
    print(f'Train data loaded: {(FT_DATASET, FT_STATE)}')
    print(f'Data shape: {X_train_tensor.shape}')
    
    # TODO: move to command line args
    # EXP_NUM = 7
    RUNTIME_LIMIT = 15
    UPDATE_LAMBDA = True
    # G_ALPHA = 0.3
    # ALG_TYPE = 'sg'
    # BATCH_SIZE = 16
    # MAXITER_GHOST = 1500
    MAXITER_ALM = np.inf
    MAXITER_SSG = np.inf
    MAXITER_SSLALM = np.inf
    TEST_SKIP_ITERS = 1
    
    read_model = False
    
    if CONSTRAINT == 'fpr':
        statistic = FalsePositiveRate()
    elif CONSTRAINT == 'pr':
        statistic = PositiveRate()
    else:
        statistic = None
    
    saved_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'saved_models'))
    directory = os.path.join(saved_models_path, DATASET_NAME,CONSTRAINT,f'{LOSS_BOUND:.0E}')
    if ALG_TYPE.startswith('sg') and not ALG_TYPE.startswith('sgd'):
        model_name = os.path.join(directory, f'{ALG_TYPE}_{LOSS_BOUND}_p{G_ALPHA}')
    else:
        model_name = os.path.join(directory, f'{ALG_TYPE}_{LOSS_BOUND}')
        
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    ftrial, ctrial, wtrial, ttrial, samples_trial = [], [], [], [], []
    
    # experiment loop
    for EXP_IDX in range(EXP_NUM):
        
        # torch.manual_seed(EXP_IDX)
        model_path = model_name + f'_trial{EXP_IDX}.pt'
        
        net = SimpleNet(in_shape=X_test.shape[1], out_shape=1, dtype=DTYPE).to(device)
        if read_model:
            net.load_state_dict(torch.load(model_path, weights_only=False, map_location=torch.device('cpu')))
        
        N = min(len(w_idx_train), len(nw_idx_train))
        
        
        if ALG_TYPE.startswith('swsg'):
            loss_fn = nn.BCEWithLogitsLoss()
            cf1 = lambda net, d: one_sided_loss_constr(loss_fn, net, d) - LOSS_BOUND
            cf2 = lambda net, d: -one_sided_loss_constr(loss_fn, net, d) - LOSS_BOUND
            c1 = FairnessConstraint(train_ds, [w_idx_train, nw_idx_train], fn=cf1, batch_size=BATCH_SIZE, seed=EXP_IDX)
            c2 = FairnessConstraint(train_ds, [w_idx_train, nw_idx_train], fn=cf2, batch_size=BATCH_SIZE, seed=EXP_IDX)
            
            alg = SSSG(net, train_ds, loss_fn, [c1, c2])
            
            history = alg.optimize(batch_size = BATCH_SIZE, epochs = epochs,
                                   save_iter = save_iter,
                                   ctol = ctol,
                                   f_stepsize_rule = f_stepsize_rule, f_stepsize = f_stepsize,
                                   c_stepsize_rule = c_stepsize_rule, c_stepsize = c_stepsize,
                                   device=device, seed=EXP_IDX, max_runtime = MAX_TIME)
            
            print(len(history['w']))
            
        elif ALG_TYPE.startswith('fairret'):
            if fconstr == 'acc':
                statistic = Accuracy()
                
            if losstype == 'norm':
                loss_fairret = NormLoss(statistic)
            elif losstype == 'lse':
                loss_fairret = LSELoss(statistic)
            elif losstype == 'kl':
                loss_fairret = KLProjectionLoss(statistic)
            run_start = timeit.default_timer()
            current_time = timeit.default_timer()
            data_w = Subset(train_ds, w_idx_train)
            data_b = Subset(train_ds, nw_idx_train)
            
            history = {'loss': [], 'constr': [], 'w': [], 'time': [], 'n_samples': []}
            loss_fn = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=5e-2)
            for epoch in range(epochs):
                gen = torch.Generator(device=device)
                gen.manual_seed(EXP_IDX+epoch)
                loader_w = torch.utils.data.DataLoader(data_w, BATCH_SIZE//2, shuffle=True, generator=gen, drop_last=True)
                loader_b = torch.utils.data.DataLoader(data_b, BATCH_SIZE//2, shuffle=True, generator=gen, drop_last=True)
                
                counter = 0
                for i, ((inputs_w, labels_w), (inputs_b, labels_b)) in enumerate(zip(loader_w, loader_b)):
                    
                    current_time = timeit.default_timer()
                    elapsed = current_time - run_start
                    if elapsed > MAX_TIME:
                        break
                    history['time'].append(elapsed)
                    history['n_samples'].append(BATCH_SIZE)
                    
                    net.zero_grad()
                    
                    inputs = torch.concat([inputs_w, inputs_b])
                    labels = torch.concat([labels_w, labels_b])
                    group_ind_onehot = torch.tensor([[0]*(BATCH_SIZE//2) + [1]*(BATCH_SIZE//2), [1]*(BATCH_SIZE//2) + [0]*(BATCH_SIZE//2)]).T
                    outputs = net(inputs)
                    loss_bce = loss_fn(outputs.squeeze(), labels)
                    counter += outputs.shape[0]
                    
                    if fconstr == 'pr':
                        loss_fr = loss_fairret(outputs.squeeze(), group_ind_onehot)
                    else:
                        loss_fr = loss_fairret(outputs, group_ind_onehot, labels.unsqueeze(1))
                    loss = loss_bce + mult*loss_fr
                    
                    loss.backward()
                    optimizer.step()
                    
                    with np.printoptions(precision=6, suppress=True):
                        print(f'{epoch:2} | {i:5} | {counter:5} | {loss_bce.detach().cpu().numpy()}|{loss_fr.detach().cpu().numpy()}', end='\r')
                    
                    history['w'].append(deepcopy(net.state_dict()))
                
        elif ALG_TYPE.startswith('sgd'):
            run_start = timeit.default_timer()
            current_time = timeit.default_timer()
            loss_fn = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=5e-3)
            train_ds = TensorDataset(X_train_tensor.to(device),y_train_tensor.to(device))
            train_l = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            history = {'loss': [], 'constr': [], 'w': [], 'time': [], 'n_samples': []}
            for epoch in range(epochs):
                for i, (inputs, labels) in enumerate(train_l):
                    elapsed = timeit.default_timer() - run_start
                    if elapsed > MAX_TIME:
                        break
                    history['time'].append(elapsed)
                    history['n_samples'].append(BATCH_SIZE)
                    
                    net.zero_grad()
                    outputs = net(inputs)
                    loss = loss_fn(outputs.squeeze(), labels)
                    loss.backward()
                    optimizer.step()
                    
                    with np.printoptions(precision=6, suppress=True):
                        print(f'{epoch:2} | {i:5} | {loss.detach().cpu().numpy()}', end='\r')
                    
                    history['w'].append(deepcopy(net.state_dict()))
                    
        elif ALG_TYPE.startswith('sg'):
            loss_fn = nn.BCEWithLogitsLoss()
            cf1 = lambda net, d: one_sided_loss_constr(loss_fn, net, d) - LOSS_BOUND
            cf2 = lambda net, d: -one_sided_loss_constr(loss_fn, net, d) - LOSS_BOUND
            c1 = FairnessConstraint(train_ds, [w_idx_train, nw_idx_train], fn=cf1, seed=EXP_IDX)
            c2 = FairnessConstraint(train_ds, [w_idx_train, nw_idx_train], fn=cf2, seed=EXP_IDX)
            
            alg = StochasticGhost(net, train_ds, loss_fn, [c1, c2])
            history = alg.optimize(geomp=G_ALPHA,
                                  stepsize_rule=ghost_stepsize_rule,
                                  zeta = ghost_zeta,
                                  gamma0 = ghost_gamma0,
                                  beta=ghost_beta,
                                  rho=ghost_rho,
                                  lamb = ghost_lambda,
                                  tau = ghost_tau,
                                  max_iter=MAXITER_GHOST,
                                  seed=EXP_IDX,
                                  max_runtime = MAX_TIME)
            
        elif ALG_TYPE.startswith('sslalm'):
            loss_fn = nn.BCEWithLogitsLoss()
            cf1 = lambda net, d: one_sided_loss_constr(loss_fn, net, d) - LOSS_BOUND
            cf2 = lambda net, d: -one_sided_loss_constr(loss_fn, net, d) - LOSS_BOUND
            c1 = FairnessConstraint(train_ds, [w_idx_train, nw_idx_train], fn=cf1, batch_size=BATCH_SIZE, seed=EXP_IDX)
            c2 = FairnessConstraint(train_ds, [w_idx_train, nw_idx_train], fn=cf2, batch_size=BATCH_SIZE, seed=EXP_IDX)
            
            alg = SSLPD(net, train_ds, loss_fn, [c1, c2])
            history = alg.optimize(batch_size=BATCH_SIZE,
                                epochs=epochs,
                                lambda_bound = 10.,
                                rho = rho,
                                mu = mu,
                                tau = tau,
                                beta = beta,
                                eta = eta,
                                max_iter=MAXITER_SSLALM,
                                device=device,
                                seed=EXP_IDX,
                                max_runtime=MAX_TIME)
            
        ## SAVE RESULTS ##
        ftrial.append(pd.Series(history['loss']))
        ctrial.append(pd.DataFrame(history['constr']))
        wtrial.append(history['w'])
        ttrial.append(history['time'])
        samples_trial.append(pd.Series(history['n_samples']))
        
        # Save the model
        
        torch.save(net.state_dict(), model_path)
        print('')
    
    # Save DataFrames to CSV files
    utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'exp_results'))
    if not os.path.exists(utils_path):
        os.makedirs(utils_path)
    
    ftrial = pd.concat(ftrial, keys=range(len(ftrial)))
    ctrial = pd.concat(ctrial, keys=range(len(ctrial)))
    samples_trial = pd.concat(samples_trial, keys=range(len(samples_trial)))
    
    if ALG_TYPE.startswith('sg') and not ALG_TYPE.startswith('sgd'):
        fname = f'{ALG_TYPE}_{DATASET_NAME}_{LOSS_BOUND}_{G_ALPHA}'
    else:
        fname = f'{ALG_TYPE}_{DATASET_NAME}_{LOSS_BOUND}'
    print(f'Saving to: {fname}')
    ftrial.to_csv(os.path.join(utils_path, fname + '_ftrial.csv'))
    ctrial.to_csv(os.path.join(utils_path, fname + '_ctrial.csv'))
    samples_trial.to_csv(os.path.join(utils_path, fname + '_samples.csv'))
    
    print('----')
    # df(n_iter, n_trials)
    wlen = max([len(tr) for tr in wtrial])
    index = pd.MultiIndex.from_product([['train', 'test'], np.arange(wlen), np.arange(EXP_NUM)], names=('is_train', 'iteration', 'trial'))
    full_stats = pd.DataFrame(index=index, columns=['Loss', 'C1', 'C2', 'SampleSize', 'time'])
    full_stats.sort_index(inplace=True)
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.set_default_device(device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    X_test_tensor = tensor(X_test, dtype=DTYPE).to(device)
    y_test_tensor = tensor(y_test, dtype=DTYPE).to(device)
    
    X_test_w = X_test_tensor[w_idx_test]
    y_test_w = y_test_tensor[w_idx_test]
    X_test_nw = X_test_tensor[nw_idx_test]
    y_test_nw = y_test_tensor[nw_idx_test]
    
    X_train_w = X_train_tensor[w_idx_train]
    y_train_w = y_train_tensor[w_idx_train]
    X_train_nw = X_train_tensor[nw_idx_train]
    y_train_nw = y_train_tensor[nw_idx_train]
    
    save_train = True
    
    
    with torch.inference_mode():
        for exp_idx in range(EXP_NUM):
            weights_to_eval = wtrial[exp_idx][::TEST_SKIP_ITERS]
            for alg_iteration, w in enumerate(weights_to_eval):
                
                if CONSTRAINT == 'loss':
                    c_f = one_sided_loss_constr
                    c_loss_fn = nn.BCEWithLogitsLoss()
                # elif CONSTRAINT == 'fpr':
                #     c_f = fairret_constr
                #     statistic = FalsePositiveRate()
                #     c_loss_fn = NormLoss(statistic)
                # elif CONSTRAINT == 'pr':
                #     c_f = fairret_pr_constr
                #     statistic = PositiveRate()
                #     c_loss_fn = NormLoss(statistic)
                print(f'{exp_idx} | {alg_iteration}', end='\r')
                # TRANSFER TO CUDA
                # net.load_state_dict([lw.to(device) for lw in w])
                net.load_state_dict(w)
                net = net.to(device)
                
                if save_train:
                    outs = net(X_train_tensor)
                    if y_train_tensor.ndim < outs.ndim:
                        y_train_tensor = y_train_tensor.unsqueeze(1)
                    loss = loss_fn(outs, y_train_tensor).detach().cpu().numpy()
                    
                    c1 = c_f(c_loss_fn, net, [(X_train_w, y_train_w), (X_train_nw, y_train_nw)]).detach().cpu().numpy()
                    c2 = -c1
                    # pandas multiindex bug(?) workaround
                    full_stats.loc['train'].at[alg_iteration, exp_idx] = {
                    'Loss': loss,
                    'C1': c1,
                    'C2': c2,
                    'SampleSize': samples_trial[exp_idx][alg_iteration],
                    'time': ttrial[exp_idx][alg_iteration]}
                    
                outs = net(X_test_tensor)
                if y_test_tensor.ndim < outs.ndim:
                    y_test_tensor = y_test_tensor.unsqueeze(1)
                loss = loss_fn(outs, y_test_tensor).detach().cpu().numpy()
                
                c1 = c_f(c_loss_fn, net, [(X_test_w, y_test_w), (X_test_nw, y_test_nw)]).detach().cpu().numpy()
                c2 = -c1
                
                full_stats.loc['test'].at[alg_iteration, exp_idx] = {
                    'Loss': loss,
                    'C1': c1,
                    'C2': c2,
                    'SampleSize': samples_trial[exp_idx][alg_iteration],
                    'time': ttrial[exp_idx][alg_iteration]}
            
    if ALG_TYPE.startswith('sg') and not ALG_TYPE.startswith('sgd'):
        fname = f'{ALG_TYPE}_{DATASET_NAME}_{LOSS_BOUND}_{G_ALPHA}.csv'
    else:
        fname = f'{ALG_TYPE}_{DATASET_NAME}_{LOSS_BOUND}.csv'
    print(f'Saving to: {fname}')
    full_stats.to_csv(os.path.join(utils_path, fname))