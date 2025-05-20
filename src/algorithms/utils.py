import torch


def net_params_to_tensor(
    net: torch.nn.Module, flatten=False, copy=False
) -> torch.Tensor:
    # flat_params = [ar.to_numpy(param) for param in net.parameters()]
    if copy:
        params = [param.detach().clone() for param in net.parameters()]
    else:
        params = [param for param in net.parameters()]

    if flatten:
        flat_params = [torch.flatten(param) for param in params]
        return torch.concat(flat_params)

    return params


def check_same_sample(sample1, sample2):
    s1w, s1nw = sample1[0], sample1[1]
    s2w, s2nw = sample2[0], sample2[1]

    s1wx, s1wy = s1w
    s2wx, s2wy = s2w

    s1nwx, s1nwy = s1nw
    s2nwx, s2nwy = s2nw

    return (
        torch.all(s1wx == s2wx)
        and torch.all(s1wy == s2wy)
        and torch.all(s1nwx == s2nwx)
        and torch.all(s1nwy == s2nwy)
    )


def net_grads_to_tensor(net, clip=False, flatten=True) -> torch.Tensor:
    param_grads = []
    if clip:
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
    for param in net.parameters():
        if param.grad is not None:
            # Clone to avoid modifying the original tensor
            if flatten:
                param_grads.append(param.grad.data.clone().view(-1))
            else:
                param_grads.append(param.grad.data.clone())
    if flatten:
        param_grads = torch.cat(param_grads)
    return param_grads


def _set_weights(net: torch.nn.Module, x):
    start = 0
    w = net_params_to_tensor(net, flatten=False, copy=False)
    for i in range(len(w)):
        end = start + w[i].numel()
        w[i].set_(x[start:end].reshape(w[i].shape))
        start = end
