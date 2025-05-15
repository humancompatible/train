import torch


def one_sided_loss_constr(loss, net, c_data):
    w_inputs, w_labels = c_data[0]
    b_inputs, b_labels = c_data[1]
    w_outs = net(w_inputs)
    if w_labels.ndim == 0:
        w_labels = w_labels.reshape(1)
        b_labels = b_labels.reshape(1)
    if w_labels.ndim < w_outs.ndim:
        w_labels = w_labels.unsqueeze(1)
        b_labels = b_labels.unsqueeze(1)
    w_loss = loss(w_outs, w_labels)
    b_outs = net(b_inputs)
    b_loss = loss(b_outs, b_labels)

    return w_loss - b_loss

def fairret_constr(loss, net, c_data):
    w_inputs, w_labels = c_data[0]
    b_inputs, b_labels = c_data[1]
    w_logits = net(w_inputs)
    b_logits = net(b_inputs)
    w_onehot = torch.tensor([[0., 1.]]*len(w_inputs))
    b_onehot = torch.tensor([[1., 0.]]*len(b_inputs))
    logits = torch.concat([w_logits, b_logits])
    sens = torch.vstack([w_onehot, b_onehot])
    labels = torch.hstack([w_labels, b_labels]).unsqueeze(1)
    # print(logits.shape)
    # print(sens.shape)
    # print(labels.shape)
    
    return loss(logits, sens, label=labels)

def fairret_pr_constr(loss, net, c_data):
    w_inputs, _ = c_data[0]
    b_inputs, _ = c_data[1]
    w_logits = net(w_inputs)
    b_logits = net(b_inputs)
    w_onehot = torch.tensor([[0., 1.]]*len(w_inputs))
    b_onehot = torch.tensor([[1., 0.]]*len(b_inputs))
    logits = torch.concat([w_logits, b_logits])
    sens = torch.vstack([w_onehot, b_onehot])
    
    return loss(logits, sens)