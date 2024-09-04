import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import StandardScaler 
from torch.nn.utils import clip_grad_norm_ 
import os


class CustomNetwork(nn.Module):

    # For now the input data is passed as init parameters
    def __init__(self, model_specs):
        super(CustomNetwork, self).__init__()

        # Create a list of linear layers based on layer_sizes
        layer_sizes = model_specs[0]
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.RANDOM_SEED = 0
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu((layer(x)))
        x = torch.sigmoid(self.layers[-1](x))
        return x
    
    

    ######Only this loss function is used here######
    def compute_loss(self, Y, Y_hat):
        L_sum = 0.5*torch.sum(torch.square(Y - Y_hat))

        m = Y.shape[0]
        # print("Y shape is: ", m)
        L = (1./m) * L_sum

        return L

    def bce_loss(self, outputs, targets):
        criterion = nn.BCELoss()
        loss = criterion(outputs, targets)
        if torch.isnan(loss).any():
            for name, param in self.named_parameters():
                print(name)
                #print(param.data)
        return loss
    
    def get_obj(self, x, y, params):
        x = torch.Tensor(x).to(torch.float32)
        y = torch.Tensor(y).to(torch.float32)
        model_parameters = list(self.parameters())
        for i in range(len(params)):
            model_parameters[i].data = torch.Tensor(params[i])
        obj_fwd = self.forward(x).flatten()
        #print("forward In OBJ: ", obj_fwd)
        if torch.isnan(obj_fwd).any():
            #print("THE MINIBATCH SIZE>>>>>",minibatch)
            for name, param in self.named_parameters():
                print(name)
                #print(param.data)
        fval = self.compute_loss(obj_fwd, y.flatten())
        return fval.item()
    
    def get_obj_grad(self, x, y, params):
        x = torch.Tensor(x).to(torch.float32)
        y = torch.Tensor(y).to(torch.float32)
        fgrad = []
        obj_fwd = self.forward(x).flatten()
        obj_loss = self.compute_loss(obj_fwd, y.flatten())
        obj_loss.backward()

        #max_norm = 0.5
        #clip_grad_norm_(self.parameters(), max_norm)
        for param in self.parameters():
            if param.grad is not None:
                # Clone to avoid modifying the original tensor
                fgrad.append(param.grad.data.clone().view(-1))

        # Manually set gradients to zero
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

        fgrad = torch.cat(fgrad, dim=0)
        return fgrad
    
    def get_constraint(self, x1_sensitive, y1_sensitive, x2_sensitive, y2_sensitive, params):
        
        x1_sensitive = torch.Tensor(x1_sensitive).to(torch.float32)
        y1_sensitive = torch.Tensor(y1_sensitive).to(torch.float32)
        x2_sensitive = torch.Tensor(x2_sensitive).to(torch.float32)
        y2_sensitive = torch.Tensor(y2_sensitive).to(torch.float32)

        sensitive_fwd1 = self.forward(x1_sensitive).flatten()
        sensitive_loss1 = self.compute_loss(sensitive_fwd1, y1_sensitive.flatten())

        sensitive_fwd2 = self.forward(x2_sensitive).flatten()
        sensitive_loss2 = self.compute_loss(sensitive_fwd2, y2_sensitive.flatten())

        cons_loss = (sensitive_loss1 - sensitive_loss2)

        return cons_loss.item()
    
    def get_constraint_grad(self, x1_sensitive, y1_sensitive, x2_sensitive, y2_sensitive, params):

        x1_sensitive = torch.Tensor(x1_sensitive).to(torch.float32)
        y1_sensitive = torch.Tensor(y1_sensitive).to(torch.float32)
        x2_sensitive = torch.Tensor(x2_sensitive).to(torch.float32)
        y2_sensitive = torch.Tensor(y2_sensitive).to(torch.float32)
        
        sensitive_fwd1 = self.forward(x1_sensitive).flatten()
        sensitive_loss1 = self.compute_loss(sensitive_fwd1, y1_sensitive.flatten())

        sensitive_fwd2 = self.forward(x2_sensitive).flatten()
        sensitive_loss2 = self.compute_loss(sensitive_fwd2, y2_sensitive.flatten())

        cons_loss = (sensitive_loss1 - sensitive_loss2)

        cgrad = []
        cons_loss.backward()
        max_norm = 0.5
        clip_grad_norm_(self.parameters(), max_norm)
        for param in self.parameters():
            if param.grad is not None:
                # Clone to avoid modifying the original tensor
                cgrad.append(param.grad.data.clone().view(-1))

        # Manually set gradients to zero
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        
        cgrad = torch.cat(cgrad, dim=0)
        return cgrad
    
    def to_backend(self, obj):
        return torch.Tensor(obj)
    
    def save_model(self, dir):
        torch.save(self, str(dir)+'.pth')

    def get_trainable_params(self):
        nn_parameters = list(self.parameters())
        initw = [param.data for param in nn_parameters]
        num_param = sum(p.numel() for p in self.parameters())
        return initw, num_param
    
    def evaluate(self, x):
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.detach().numpy()
    
def load_model(directory_path, model_file):
        model = torch.load(os.path.join(directory_path, model_file), map_location=torch.device('cpu'))
        return model

