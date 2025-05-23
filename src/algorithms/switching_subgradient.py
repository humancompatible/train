import timeit
from copy import deepcopy
from typing import Callable

import numpy as np
import torch

from src.algorithms.Algorithm import Algorithm
from src.algorithms.utils import net_params_to_tensor


class SSG(Algorithm):
    def __init__(
        self, net, data, loss, constraints, custom_project_fn: Callable = None
    ):
        super().__init__(net, data, loss, constraints)
        self.project = custom_project_fn if custom_project_fn else self.project_fn

    @staticmethod
    def project_fn(x, m):
        return x

    def optimize(
        self,
        ctol,
        f_stepsize_rule,
        f_stepsize,
        c_stepsize_rule,
        c_stepsize,
        batch_size,
        epochs,
        save_iter=None,
        device="cpu",
        seed=None,
        verbose=True,
        max_runtime=None,
        max_iter=None,
    ):
        run_start = timeit.default_timer()

        f_eta_t = f_stepsize
        c_eta_t = c_stepsize

        loss_eval = None
        c_t = None
        eta_f_sum = total_iters = iteration = f_iters = c_iters = epoch = 0
        eta_f_list = []
        _ctol = ctol
        if epochs is None:
            epochs = np.inf
        if max_iter is None:
            max_iter = np.inf

        gen = torch.Generator(device=device)
        loss_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size, shuffle=True, generator=gen
        )
        loss_iter = iter(loss_loader)

        while True:
            elapsed = timeit.default_timer() - run_start
            iteration += 1
            total_iters += 1
            if epoch >= epochs or total_iters >= max_iter or elapsed > max_runtime:
                break

            self.history["w"].append(deepcopy(self.net.state_dict()))
            self.history["time"].append(elapsed)
            self.history["n_samples"].append(batch_size * 3)

            try:
                f_sample = next(loss_iter)
            except StopIteration:
                epoch += 1
                f_iters = 0
                c_iters = 0
                iteration = 0
                _ctol = ctol
                loss_loader = torch.utils.data.DataLoader(
                    self.dataset, batch_size, shuffle=True, generator=gen
                )
                loss_iter = iter(loss_loader)
                f_sample = next(loss_iter)

            self.net.zero_grad()
            if iteration > 800:
                _ctol *= 0.97

            if save_iter is not None and total_iters >= save_iter:
                eta_f_list.append(f_eta_t)
                eta_f_sum += f_eta_t

            # generate sample of constraints
            c_sample = [ci.sample_loader() for ci in self.constraints]
            # calc constraints and update multipliers (line 3)
            with torch.no_grad():
                c_t = torch.concat(
                    [
                        ci.eval(self.net, c_sample[i]).reshape(1)
                        for i, ci in enumerate(self.constraints)
                    ]
                )
                c_max = torch.max(c_t)
            self.history["constr"].append(c_max.cpu().detach().numpy())

            x_t = net_params_to_tensor(self.net, flatten=True, copy=True)

            if c_max >= _ctol:
                c_iters += 1

                self.history["n_samples"].append(batch_size * 2)
                # calculate grad on an independent sample
                c_sample = [ci.sample_loader() for ci in self.constraints]
                c_t2 = torch.concat(
                    [
                        ci.eval(self.net, c_sample[i]).reshape(1)
                        for i, ci in enumerate(self.constraints)
                    ]
                )
                c_max2 = torch.max(c_t2)

                c_grad = torch.autograd.grad(c_max2, self.net.parameters())
                c_grad = torch.concat([cg.flatten() for cg in c_grad])

                if c_stepsize_rule == "adaptive":
                    c_eta_t = c_max / torch.norm(c_grad) ** 2
                elif c_stepsize_rule == "const":
                    c_eta_t = c_stepsize
                elif c_stepsize_rule == "dimin":
                    c_eta_t = c_stepsize / np.sqrt(c_iters)

                x_t1 = self.project(x_t - c_eta_t * c_grad, m=2)

            else:
                self.history["n_samples"].append(batch_size)
                f_iters += 1
                f_inputs, f_labels = f_sample
                outputs = self.net(f_inputs)
                if f_labels.dim() < outputs.dim():
                    f_labels = f_labels.unsqueeze(1)
                loss_eval = self.loss_fn(outputs, f_labels)
                self.history["loss"].append(loss_eval.cpu().detach().numpy())

                f_grad = torch.autograd.grad(loss_eval, self.net.parameters())
                f_grad = torch.concat([fg.flatten() for fg in f_grad])

                if f_stepsize_rule == "dimin":
                    f_eta_t = f_stepsize / np.sqrt(f_iters)
                elif f_stepsize_rule == "const":
                    f_eta_t = f_stepsize
                x_t1 = self.project(x_t - f_eta_t * f_grad, m=2)

            start = 0
            with torch.no_grad():
                w = net_params_to_tensor(self.net, flatten=False, copy=False)
                for i in range(len(w)):
                    end = start + w[i].numel()
                    w[i].set_(x_t1[start:end].reshape(w[i].shape))
                    start = end

            if verbose and loss_eval is not None and c_t is not None:
                with np.printoptions(precision=6, suppress=True):
                    print(
                        f"{epoch:2} | {iteration:5} |{_ctol:.5}|{loss_eval.detach().cpu().numpy()}|{c_t.detach().cpu().numpy()}",
                        end="\r",
                    )

        ######################
        ### POSTPROCESSING ###
        ######################

        if save_iter is not None:
            model_ind = np.random.default_rng(seed=seed).choice(
                np.arange(start=save_iter, stop=total_iters),
                p=np.array(eta_f_list) / np.sum(eta_f_list),
            )
            self.net.load_state_dict(self.history["w"][model_ind])

        return self.history
