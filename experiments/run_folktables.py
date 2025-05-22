from copy import deepcopy
import importlib
import os
import timeit

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, tensor
from torch.utils.data import TensorDataset, Subset
from utils.load_folktables import prepare_folktables
from utils.network import SimpleNet

from src.constraints import FairnessConstraint


@hydra.main(version_base=None, config_path="conf", config_name="experiment")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    N_RUNS = cfg.n_runs
    FT_STATE = cfg.data.state
    FT_TASK = cfg.data.task
    DOWNLOAD_DATA = cfg.data.download
    DATA_PATH = cfg.data.path
    
    # CONSTRAINT = cfg.constraint
    CONSTRAINT = 'eq_loss'
    LOSS_BOUND = cfg.constraint.bound

    if cfg.device == "cpu":
        device = "cpu"
    elif cfg.alg.startswith("sg"):
        device = "cpu"
        print("CUDA not supported for Stochastic Ghost")
    elif torch.cuda.is_available():
        device = "cuda"
        print("CUDA found")
    else:
        device = "cpu"
        print("CUDA not found")

    print(f"{device = }")
    torch.set_default_device(device)

    DTYPE = torch.float32

    torch.set_default_dtype(DTYPE)
    DATASET_NAME = FT_TASK + "_" + FT_STATE

    (
        X_train,
        y_train,
        [w_idx_train, nw_idx_train],
        X_test,
        y_test,
        [w_idx_test, nw_idx_test],
    ) = prepare_folktables(
        FT_TASK,
        state=FT_STATE.upper(),
        random_state=42,
        make_unbalanced=False,
        onehot=False,
        download=DOWNLOAD_DATA,
        path=DATA_PATH,
    )
    X_train_tensor = tensor(X_train, dtype=DTYPE)
    y_train_tensor = tensor(y_train, dtype=DTYPE)
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    print(f"Train data loaded: {(FT_TASK, FT_STATE)}")
    print(f"Data shape: {X_train_tensor.shape}")

    saved_models_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "utils", "saved_models")
    )
    directory = os.path.join(
        saved_models_path, DATASET_NAME, CONSTRAINT, f"{LOSS_BOUND:.0E}"
    )

    model_name = os.path.join(directory, f"{cfg.alg.import_name}_{LOSS_BOUND}")

    if not os.path.exists(directory):
        os.makedirs(directory)

    ftrial, ctrial, wtrial, ttrial, samples_trial = [], [], [], [], []

    # experiment loop
    for EXP_IDX in range(N_RUNS):
        torch.manual_seed(EXP_IDX)
        model_path = model_name + f"_trial{EXP_IDX}.pt"

        net = SimpleNet(in_shape=X_test.shape[1], out_shape=1, dtype=DTYPE).to(device)
        
        if cfg.alg.import_name.startswith("fairret"):
            fairret_loss = importlib.import_module("fairret.loss")
            fairret_statistic = importlib.import_module("fairret.statistic")
            statistic = getattr(fairret_statistic, cfg.alg.params.statistic)()
            loss_fairret = getattr(fairret_loss, cfg.alg.params.loss)(statistic)

            run_start = timeit.default_timer()
            current_time = timeit.default_timer()
            data_w = Subset(train_ds, w_idx_train)
            data_b = Subset(train_ds, nw_idx_train)

            history = {"loss": [], "constr": [], "w": [], "time": [], "n_samples": []}
            loss_fn = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=5e-2)
            epochs = cfg.alg.params.epochs
            batch_size = cfg.alg.params.batch_size
            mult = cfg.alg.params.pmult
            for epoch in range(epochs):
                gen = torch.Generator(device=device)
                gen.manual_seed(EXP_IDX + epoch)
                loader_w = torch.utils.data.DataLoader(
                    data_w, batch_size // 2, shuffle=True, generator=gen, drop_last=True
                )
                loader_b = torch.utils.data.DataLoader(
                    data_b, batch_size // 2, shuffle=True, generator=gen, drop_last=True
                )
                for i, ((inputs_w, labels_w), (inputs_b, labels_b)) in enumerate(
                    zip(loader_w, loader_b)
                ):
                    current_time = timeit.default_timer()
                    elapsed = current_time - run_start
                    if elapsed > cfg.run_maxtime:
                        break
                    history["time"].append(elapsed)
                    history["n_samples"].append(batch_size)

                    net.zero_grad()

                    inputs = torch.concat([inputs_w, inputs_b])
                    labels = torch.concat([labels_w, labels_b])
                    group_ind_onehot = torch.tensor(
                        [
                            [0] * (batch_size // 2) + [1] * (batch_size // 2),
                            [1] * (batch_size // 2) + [0] * (batch_size // 2),
                        ]
                    ).T
                    outputs = net(inputs)
                    loss_bce = loss_fn(outputs.squeeze(), labels)

                    try:
                        loss_fr = loss_fairret(outputs.squeeze(), group_ind_onehot)
                    except:
                        loss_fr = loss_fairret(
                            outputs, group_ind_onehot, labels.unsqueeze(1)
                        )
                    loss = loss_bce + mult * loss_fr

                    loss.backward()
                    optimizer.step()

                    with np.printoptions(precision=6, suppress=True):
                        print(
                            f"{epoch:2} | {i:5} | {loss_bce.detach().cpu().numpy():.4}|{loss_fr.detach().cpu().numpy():.4}",
                            end="\r",
                        )

                    history["w"].append(deepcopy(net.state_dict()))

        elif cfg.alg.import_name.lower().startswith("sgd"):
            run_start = timeit.default_timer()
            current_time = timeit.default_timer()
            loss_fn = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=cfg.alg.params.lr)
            train_ds = TensorDataset(
                X_train_tensor.to(device), y_train_tensor.to(device)
            )
            batch_size = cfg.alg.params.batch_size
            train_l = torch.utils.data.DataLoader(
                train_ds, batch_size=batch_size, shuffle=True
            )
            history = {"loss": [], "constr": [], "w": [], "time": [], "n_samples": []}
            for epoch in range(cfg.alg.params.epochs):
                for i, (inputs, labels) in enumerate(train_l):
                    elapsed = timeit.default_timer() - run_start
                    if elapsed > cfg.run_maxtime:
                        break
                    history["time"].append(elapsed)
                    history["n_samples"].append(batch_size)

                    net.zero_grad()
                    outputs = net(inputs)
                    loss = loss_fn(outputs.squeeze(), labels)
                    loss.backward()
                    optimizer.step()

                    with np.printoptions(precision=6, suppress=True):
                        print(
                            f"{epoch:2} | {i:5} | {loss.detach().cpu().numpy()}",
                            end="\r",
                        )

                    history["w"].append(deepcopy(net.state_dict()))
        else:
            constraint_fn_module = importlib.import_module("src.constraints")
            constraint_fn = getattr(constraint_fn_module, cfg.constraint.import_name)

            loss_fn = nn.BCEWithLogitsLoss()
            cf1 = lambda net, d: constraint_fn(loss_fn, net, d) - cfg.constraint.bound
            cf2 = (
                lambda net, d: -constraint_fn(loss_fn, net, d) - cfg.constraint.bound
            )
            c1 = FairnessConstraint(
                train_ds,
                [w_idx_train, nw_idx_train],
                fn=cf1,
                batch_size=cfg.constraint.c_batch_size,
                seed=EXP_IDX,
            )
            c2 = FairnessConstraint(
                train_ds,
                [w_idx_train, nw_idx_train],
                fn=cf2,
                batch_size=cfg.constraint.c_batch_size,
                seed=EXP_IDX,
            )
            
            optimizer_name = cfg.alg.import_name
            module = importlib.import_module("src.algorithms")
            Optimizer = getattr(module, optimizer_name)

            optimizer = Optimizer(net, train_ds, loss_fn, [c1, c2])
            history = optimizer.optimize(
                **cfg.alg.params,
                max_iter=cfg.run_maxiter,
                max_runtime=cfg.run_maxtime,
                device=cfg.device,
                seed=EXP_IDX,
            )

        ## SAVE RESULTS ##
        ftrial.append(pd.Series(history["loss"]))
        ctrial.append(pd.DataFrame(history["constr"]))
        wtrial.append(history["w"])
        ttrial.append(history["time"])
        samples_trial.append(pd.Series(history["n_samples"]))

        ## SAVE MODEL ##
        torch.save(net.state_dict(), model_path)
        print("")

    # Save DataFrames to CSV files
    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "utils", "exp_results")
    )
    if not os.path.exists(utils_path):
        os.makedirs(utils_path)

    ftrial = pd.concat(ftrial, keys=range(len(ftrial)))
    ctrial = pd.concat(ctrial, keys=range(len(ctrial)))
    samples_trial = pd.concat(samples_trial, keys=range(len(samples_trial)))

    fname = f"{cfg.alg.import_name}_{DATASET_NAME}_{LOSS_BOUND}"

    print(f"Saving to: {fname}")
    ftrial.to_csv(os.path.join(utils_path, fname + "_ftrial.csv"))
    ctrial.to_csv(os.path.join(utils_path, fname + "_ctrial.csv"))
    samples_trial.to_csv(os.path.join(utils_path, fname + "_samples.csv"))
    print("Saved!")

    #############################################################
    ### CALCULATE TEST SET STATS ON EVERY ALGORITHM ITERATION ###
    #############################################################

    print("----")
    print("")
    wlen = max([len(tr) for tr in wtrial])
    index = pd.MultiIndex.from_product(
        [["train", "test"], np.arange(wlen), np.arange(N_RUNS)],
        names=("is_train", "iteration", "trial"),
    )
    full_stats = pd.DataFrame(
        index=index, columns=["Loss", "C1", "C2", "SampleSize", "time"]
    )
    full_stats.sort_index(inplace=True)

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
        for exp_idx in range(N_RUNS):
            weights_to_eval = wtrial[exp_idx]
            for alg_iteration, w in enumerate(weights_to_eval):
                if CONSTRAINT == "eq_loss":
                    constraint_fn_module = importlib.import_module("src.constraints")
                    constraint_fn = getattr(constraint_fn_module, cfg.constraint.import_name)
                    c_f = constraint_fn
                    c_loss_fn = nn.BCEWithLogitsLoss()
                print(f"{exp_idx} | {alg_iteration}", end="\r")
                net.load_state_dict(w)
                net = net.to(device)

                if save_train:
                    outs = net(X_train_tensor)
                    if y_train_tensor.ndim < outs.ndim:
                        y_train_tensor = y_train_tensor.unsqueeze(1)
                    loss = loss_fn(outs, y_train_tensor).detach().cpu().numpy()

                    c1 = (
                        c_f(
                            c_loss_fn,
                            net,
                            [(X_train_w, y_train_w), (X_train_nw, y_train_nw)],
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    c2 = -c1
                    # pandas multiindex bug(?) workaround
                    full_stats.loc["train"].at[alg_iteration, exp_idx] = {
                        "Loss": loss,
                        "C1": c1,
                        "C2": c2,
                        "SampleSize": samples_trial[exp_idx][alg_iteration],
                        "time": ttrial[exp_idx][alg_iteration],
                    }

                outs = net(X_test_tensor)
                if y_test_tensor.ndim < outs.ndim:
                    y_test_tensor = y_test_tensor.unsqueeze(1)
                loss = loss_fn(outs, y_test_tensor).detach().cpu().numpy()

                c1 = (
                    c_f(c_loss_fn, net, [(X_test_w, y_test_w), (X_test_nw, y_test_nw)])
                    .detach()
                    .cpu()
                    .numpy()
                )
                c2 = -c1

                full_stats.loc["test"].at[alg_iteration, exp_idx] = {
                    "Loss": loss,
                    "C1": c1,
                    "C2": c2,
                    "SampleSize": samples_trial[exp_idx][alg_iteration],
                    "time": ttrial[exp_idx][alg_iteration],
                }

    fname = f"{cfg.alg.import_name}_{DATASET_NAME}_{LOSS_BOUND}.csv"
    print(f"Saving to: {fname}")
    full_stats.to_csv(os.path.join(utils_path, fname))


if __name__ == "__main__":
    run()
