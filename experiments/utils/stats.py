import numpy as np
import ot
import pandas as pd
import torch
from fairret.statistic import *
from sklearn.metrics import auc, roc_curve

from src.constraints.constraint_fns import *


def fair_stats(p_1, y_1, p_2, y_2):
    '''
    Compute Independence, Separation, Inaccuracy, Sufficiency.
    '''
    p = torch.concat([torch.tensor(p_1), torch.tensor(p_2)]).unsqueeze(1)
    w_onehot = torch.tensor([[0.0, 1.0]] * len(p_1))
    b_onehot = torch.tensor([[1.0, 0.0]] * len(p_2))
    sens = torch.vstack([w_onehot, b_onehot])
    labels = torch.concat([torch.tensor(y_1), torch.tensor(y_2)]).unsqueeze(1)
    pr0, pr1 = PositiveRate()(p, sens)
    fpr0, fpr1 = FalsePositiveRate()(p, sens, labels)
    tpr0, tpr1 = TruePositiveRate()(p, sens, labels)
    tnr0, tnr1 = 1 - fpr0, 1 - fpr1
    fnr0, fnr1 = 1 - tpr0, 1 - tpr1
    acc0, acc1 = Accuracy()(p, sens, labels)
    ppv0, ppv1 = PositivePredictiveValue()(p, sens, labels)
    fomr0, fomr1 = FalseOmissionRate()(p, sens, labels)
    npv0, npv1 = 1 - fomr0, 1 - fomr1

    ind = abs(pr0 - pr1)
    sp = abs(tpr0 - tpr1) + abs(fpr0 - fpr1)

    ina = sum(np.concatenate([p_1, p_2]) != np.concatenate([y_1, y_2])) / (
        len(p_1) + len(p_2)
    )
    sf = abs(ppv0 - ppv1) + abs(npv0 - npv1)
    return ind, sp, ina, sf


@torch.inference_mode()
def make_model_stats_table(X_w, y_w, X_nw, y_nw, loaded_models):
    results_list = []
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for model_index, model_iter in enumerate(loaded_models):
        (model_name, model) = model_iter

        # else:
        alg = model_name
        predictions_0 = model(X_w)
        predictions_1 = model(X_nw)
        if torch.any(torch.isnan(predictions_0)) or torch.any(
            torch.isnan(predictions_1)
        ):
            print(f"skipped {model_name}")
            continue
        y_w = y_w.squeeze()
        y_nw = y_nw.squeeze()
        l_0 = loss_fn(predictions_0[:, 0], y_w).cpu().numpy()
        l_1 = loss_fn(predictions_1[:, 0], y_nw).cpu().numpy()
        predictions_0 = torch.nn.functional.sigmoid(predictions_0[:, 0])
        predictions_1 = torch.nn.functional.sigmoid(predictions_1[:, 0])
        # Calculate AUCs for sensitive attribute 0
        fpr_0, tpr_0, thresholds_0 = roc_curve(
            y_w.cpu().numpy(), predictions_0.cpu().numpy()
        )
        auc_0 = auc(fpr_0, tpr_0)
        # Calculate AUCs for sensitive attribute 1
        fpr_1, tpr_1, thresholds_1 = roc_curve(
            y_nw.cpu().numpy(), predictions_1.cpu().numpy()
        )
        auc_1 = auc(fpr_1, tpr_1)
        auc_hm = (auc_0 * auc_1) / (auc_0 + auc_1)
        auc_m = (auc_0 + auc_1) / 2
        # Calculate TPR-FPR difference for sensitive attribute 0
        tpr_minus_fpr_0 = tpr_0 - fpr_0
        optimal_threshold_index_0 = np.argmax(tpr_minus_fpr_0)
        optimal_threshold_0 = thresholds_0[optimal_threshold_index_0]

        # Calculate TPR-FPR difference for sensitive attribute 1
        tpr_minus_fpr_1 = tpr_1 - fpr_1
        optimal_threshold_index_1 = np.argmax(tpr_minus_fpr_1)
        optimal_threshold_1 = thresholds_1[optimal_threshold_index_1]

        p_0_np = (predictions_0 > 0.5).cpu().numpy()
        p_1_np = (predictions_1 > 0.5).cpu().numpy()
        y_w_np = y_w.cpu().numpy()
        y_nw_np = y_nw.cpu().numpy()

        ind, sp, ina, sf = fair_stats(p_0_np, y_w_np, p_1_np, y_nw_np)

        a0, x0 = np.histogram(predictions_0, bins=50)
        a1, x1 = np.histogram(predictions_1, bins=x0)
        a0 = a0.astype(float)
        a1 = a1.astype(float)
        a0 /= np.sum(a0)
        a1 /= np.sum(a1)
        wd = ot.wasserstein_1d(x0[1:], x1[1:], a0, a1, p=2)
        # Store results in the DataFrame
        results_list.append(
            {
                "Model": str(model_name),
                "Algorithm": alg,
                "AUC_M": auc_m,
                "Ind": ind,
                "Sp": sp,
                "Ina": ina,
                "Sf": sf,
                "Wd": wd,
                "|Loss_0 - Loss_1|": abs(l_0 - l_1),
            }
        )

    res_df = pd.DataFrame(results_list)
    return res_df

# def get_alg_name(alg: str):
#     if alg.startswith("swsg"):
#         return "Switching Subgradient"
#     elif alg.startswith("sgd"):
#         return "SGD"
#     elif alg.startswith("sg"):
#         return "Stochastic Ghost"
#     elif alg.startswith("sslalm_mu0"):
#         return "ALM"
#     elif alg.startswith("sslalm"):
#         return "SSL-ALM"
#     elif alg.startswith("fairret"):
#         return "SGD + Fairret"

def aggregate_model_stats_table(table: pd.DataFrame, agg_fns):
    if len(agg_fns) == 1 and not isinstance(agg_fns, str):
        df = (
            table.drop("Model", axis=1)
            .groupby("Algorithm")
            .agg(agg_fns[0])
            .sort_index()
        )
    else:
        df = table.drop("Model", axis=1).groupby("Algorithm").agg(agg_fns)

    df["Algname"] = df.apply(lambda row: row.name, axis=1)
    df["Algname"] = pd.Categorical(
        df["Algname"],
        [
            "SGD",
            "SGD + Fairret",
            "Stochastic Ghost",
            "ALM",
            "SSL-ALM",
            "Switching Subgradient",
        ],
    )
    df = df.sort_values(by="Algname", axis=0)
    return df