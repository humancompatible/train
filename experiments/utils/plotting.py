import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def getRoundedThresholdv1(a, round_step):
    return np.round(a / round_step) * round_step


def plot_iter(
    data,
    lb,
    x_axis="iteration",
    q1=0.25,
    q2=0.75,
    f_ylim=(0, 0.75),
    c_ylim=(-0.01, 0.01),
):  # save=False, dataset_name=None):
    q1 = q1
    q2 = q2
    q3 = 0.5

    means = data.groupby(x_axis).mean()
    q_lower = data.groupby(by=x_axis).quantile(q=q1, interpolation="lower")
    q_mid = data.groupby(by=x_axis).quantile(q=q3, interpolation="linear")
    q_higher = data.groupby(by=x_axis).quantile(q=q2, interpolation="higher")

    f = plt.figure()

    ax1 = f.add_subplot()

    ax1.fill_between(x=means.index, y1=q_lower["Loss"], y2=q_higher["Loss"], alpha=0.4)
    ax1.plot(q_lower["Loss"], label=f"Q{int(q1 * 100)}", c="black", lw=0.6)
    ax1.plot(q_higher["Loss"], label=f"Q{int(q2 * 100)}", c="black", lw=0.6)
    ax1.plot(q_mid["Loss"], label="Median", c="darkorange", lw=0.6)
    ax1.plot(means["Loss"], label="Mean")
    xt = ax1.get_xticks()
    xt_ind = xt[1:-1] - 1
    xt_ind[0] = 0
    # ax1.set_xticks(means['SampleSize'].cumsum()[xt_ind])
    # ax1.set_xticklabels(labels=np.round(means['SampleSize'].cumsum()[xt_ind], 0), rotation=45)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # if save:
    # f.savefig('C:/Users/andre/docs/plots/sslalm/income_race/loss

    f_ = plt.figure()
    ax2 = f_.add_subplot()

    ax2.fill_between(x=means.index, y1=q_lower["C1"], y2=q_higher["C1"], alpha=0.4)
    ax2.plot(q_lower["C1"], ls="-", label=f"Q{int(q1 * 100)}", c="black", lw=0.6)
    ax2.plot(q_higher["C1"], ls="-", label=f"Q{int(q2 * 100)}", c="black", lw=0.6)
    ax2.plot(q_mid["C1"], label="Median", c="darkorange")
    ax2.plot(means["C1"], label="Mean")

    ax2.set_xlabel("Iteration")
    # ax2.set_ylim(bottom=-0.02, top=0.02)
    ax2.hlines(
        y=[-lb, lb],
        xmin=0,
        xmax=max(data["iteration"]),
        ls="--",
        colors="blue",
        alpha=0.5,
        label="Constraint bound",
    )
    ax2.hlines(
        y=0, xmin=0, xmax=max(data["iteration"]), ls="--", colors="black", alpha=0.5
    )
    ax2.set_ylabel("$L_w-L_b$")
    ax2.legend()
    return f, f_


def plot_trajectories(data, lb, x_axis, alpha=0.5, lw=1, legend=True):
    f = plt.figure()
    ax1 = f.add_subplot()
    f = plt.figure()
    ax2 = f.add_subplot()
    for EXP_NUM in data["trial"].unique():
        traj = data[data["trial"] == EXP_NUM]
        if x_axis == "time":
            x = traj["time"]
        elif x_axis == "iteration":
            x = traj["iteration"]
        if isinstance(alpha, list):
            _a = alpha[EXP_NUM]
        else:
            _a = alpha
        if _a == 0:
            continue
        ax1.plot(x, traj["Loss"], label="Loss - trial {EXP_NUM}", alpha=_a, lw=lw)
        ax2.plot(x, traj["C1"], label=f"C1 - trial {EXP_NUM}", alpha=_a, lw=lw)

    ax1.set_xlabel("iteration" if x_axis == "iteration" else "time, s")
    # ax1.set_ybound(0, 1)
    ax2.set_xlabel("iteration" if x_axis == "iteration" else "time, s")
    ax2.hlines(
        y=[-lb, lb],
        xmin=0,
        xmax=max(data[x_axis]),
        ls="--",
        colors="blue",
        alpha=0.5,
        label="Constraint bound",
    )
    ax2.hlines(y=0, xmin=0, xmax=max(data[x_axis]), ls="--", colors="black", alpha=0.5)
    # ax2.set_ybound(-0.02, 0.04)
    ax2.set_ylabel("$L_w-L_b$")
    if legend:
        ax2.legend()
    # f.show()


def plot_time(
    data,
    lb,
    round_step=0.5,
    fill="bfill",
    fill_limit=None,
    q1=0.25,
    q2=0.75,
    f_ylim=(0.4, 0.75),
    c_ylim=(-0.06, 0.07),
):
    q3 = 0.5

    data["time_r"] = getRoundedThresholdv1(data["time"], round_step)

    time_step_idx = pd.Index(np.arange(0, max(data["time_r"]), step=round_step))

    trials = []

    for EXP_NUM in data["trial"].unique():
        trial_stats = data[data["trial"] == EXP_NUM]
        trial_stats.index = trial_stats["time_r"]
        trial_stats = trial_stats.reindex(time_step_idx, copy=True)
        trial_stats["time_r"] = trial_stats.index
        if fill == "bfill":
            trial_stats.bfill(inplace=True, limit=fill_limit)
        elif fill == "ffill":
            trial_stats.ffill(inplace=True, limit=fill_limit)
        else:
            trial_stats.interpolate(fill, inplace=True, limit_direction="forward")
        trials.append(trial_stats)

    trials = pd.concat(trials, ignore_index=True)
    trials_gr = trials.groupby("time_r")

    # f, axs = plt.subplots(1,5)
    # for EXP_NUM in data['trial'].unique():
    #     axs[EXP_NUM].set_title(EXP_NUM)
    #     tr = trials[trials['trial'] == EXP_NUM]
    #     axs[EXP_NUM].plot(tr['time_r'], tr['Loss'])

    means = trials_gr.mean()
    q_lower = trials_gr.quantile(q=q1, interpolation="lower")
    q_mid = trials_gr.quantile(q=q3, interpolation="linear")
    q_higher = trials_gr.quantile(q=q2, interpolation="higher")

    f = plt.figure()

    ax1 = f.add_subplot()

    ax1.fill_between(x=means.index, y1=q_lower["Loss"], y2=q_higher["Loss"], alpha=0.4)
    ax1.plot(q_lower["Loss"], label=f"Q{int(q1 * 100)}", c="black", lw=0.6)
    ax1.plot(q_higher["Loss"], label=f"Q{int(q2 * 100)}", c="black", lw=0.6)
    ax1.plot(q_mid["Loss"], label="Median", c="darkorange")
    ax1.plot(means["Loss"], label="Mean")
    ax1.set_ylim(bottom=f_ylim[0], top=f_ylim[1])

    xt = ax1.get_xticks()
    xt_ind = xt[1:-1] - 1
    xt_ind[0] = 0
    # ax1.set_xticks(means['SampleSize'].cumsum()[xt_ind])
    # ax1.set_xticklabels(labels=np.round(means['SampleSize'].cumsum()[xt_ind], 0), rotation=45)

    ax1.set_xlabel("time, s")
    ax1.set_ylabel("Loss")
    ax1.legend()

    f_ = plt.figure()
    ax2 = f_.add_subplot()

    ax2.fill_between(x=means.index, y1=q_lower["C1"], y2=q_higher["C1"], alpha=0.4)
    ax2.plot(q_lower["C1"], ls="-", label=f"Q{int(q1 * 100)}", c="black", lw=0.6)
    ax2.plot(q_higher["C1"], ls="-", label=f"Q{int(q2 * 100)}", c="black", lw=0.6)
    ax2.plot(q_mid["C1"], label="Median", c="darkorange")
    ax2.plot(means["C1"], label="Mean")

    ax2.set_xlabel("time, s")
    # ax2.set_ylim(bottom=-0.02, top=0.02)
    ax2.hlines(
        y=[-lb, lb],
        xmin=0,
        xmax=max(means.index),
        ls="--",
        colors="blue",
        alpha=0.5,
        label="Constraint bound",
    )
    ax2.hlines(y=0, xmin=0, xmax=max(means.index), ls="--", colors="black", alpha=0.5)
    ax2.set_ylabel("$L_w-L_b$")
    ax2.legend()
    ax2.set_ylim(bottom=c_ylim[0], top=c_ylim[1])

    return f, f_




def spider_line(data, title=None):
    plt.rcParams.update({"font.size": 16})

    labels = ["Ind", "Sep", "Ina", "Suf"]
    # Number of variables we're plotting.
    num_vars = len(labels)

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    angles += angles[:1]
    labels += labels[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for alg in data.index:
        values = data.loc[alg, ["Ind", "Sp", "Ina", "Sf", "Ind"]].tolist()
        ax.plot(angles, values, lw=2, label=alg)
        # ax.plot(angles, values, lw=2, label=alg)
        ax.set_yticks([0, 0.1, 0.2, 0.3])

    plt.thetagrids(np.degrees(angles), labels=labels)
    if title:
        ax.set_title(title)
    fig.legend()
    fig.tight_layout()
    return fig