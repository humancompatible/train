import pandas as pd
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
import importlib

def printmd(string):
    display(Markdown(string))
        
from sklearn.preprocessing import StandardScaler 
import argparse

sys.path.append("..")  # Add parent directory to the sys.path

#import StochasticGhost

from folktables import ACSDataSource, ACSPublicCoverage


RACE_IND = 4
SENSITIVE_CODE_0 = 0
SENSITIVE_CODE_1 = 1
SENSITIVE_CODE_2 = 2

def preprocess_data():

    data_source = ACSDataSource(survey_year=2018, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)

    ca_features, ca_labels, _ = ACSPublicCoverage.df_to_pandas(acs_data)

    ca_features_filt = ca_features[(ca_features['RAC1P'] == 1) | (ca_features['RAC1P'] == 2) | (ca_features['RAC1P'] == 6)]
    ca_labels_filt = ca_labels[(ca_features['RAC1P'] == 1) | (ca_features['RAC1P'] == 2) | (ca_features['RAC1P'] == 6)]
    ca_features_filt['RAC1P'] = ca_features_filt['RAC1P'].replace({1: 0, 2: 2, 6: 1})
    ca_features_filt['SEX'] = ca_features_filt['SEX'].replace({1: 0, 2: 1})

    # filterning based on non-numeric values
    ca_features_filt[ca_features_filt.select_dtypes(include='object').any(axis=1)]
    ca_labels_filt[ca_features_filt.select_dtypes(include='object').any(axis=1)]

    ###### Create coverage bins
    pincp_column = ca_features_filt["PINCP"]
    # Calculate bin size
    bin_size = (pincp_column.max() - pincp_column.min()) / 10
    # Create bins
    bins = np.arange(pincp_column.min(), pincp_column.max() + bin_size, bin_size)
    # Assign values to bins
    pincp_bins = pd.cut(pincp_column, bins=bins, labels=False)
    # Add new column
    ca_features_filt["PINCP_cat"] = pincp_bins

    ###### Create age bins
    # Assuming 'ca_features_filt' is your DataFrame
    age_column = ca_features_filt["AGEP"]
    # Calculate bin size
    bin_size = (age_column.max() - age_column.min()) / 5
    # Create bins
    bins = np.arange(age_column.min(), age_column.max() + bin_size, bin_size)
    # Assign values to bins
    age_bins = pd.cut(age_column, bins=bins, labels=False)
    # Add new column
    ca_features_filt["AGEP_cat"] = age_bins

    ###### Output label to int
    ca_labels_filt['PUBCOV'] = ca_labels_filt['PUBCOV'].astype(int)

    #print(ca_features_filt["AGEP_cat"])

    # Get the indices of rows with no NaNs in ca_features_filt
    valid_indices = ca_features_filt.dropna().index

    # Filter ca_labels_filt based on valid indices
    ca_features_filt = ca_features_filt.dropna()
    ca_labels_filt = ca_labels_filt.loc[valid_indices]

    in_cols = ["AGEP_cat", "SEX", "SCHL", "MAR", "RAC1P", "DIS", "CIT", "MIG", "DEAR", "DEYE", "DREM", "PINCP_cat", "FER"]
    out_cols = ["PUBCOV"]
    x_train, x_val, y_train, y_val = train_test_split(ca_features_filt[in_cols].values, ca_labels_filt[out_cols].values, test_size  = 0.30)

    # Normalization
    scaler = StandardScaler()  
    # Fitting only on training data
    scaler.fit(x_train)  
    X_train = scaler.transform(x_train)  
    # Applying same transformation to test data
    X_val = scaler.transform(x_val)

    
    file_path_raw = os.path.join('../data/val_data', 'val_data_raw_coverage.csv')
    file_path_scaled = os.path.join('../data/val_data', 'val_data_scaled_coverage.csv')

    data_combined_raw = np.concatenate((x_val, y_val), axis=1)

    # Convert the combined data to a DataFrame
    df_combined = pd.DataFrame(data_combined_raw)

    #x_val_columns = ['priors_count', 'score_code', 'age_code', 'gender_code', 'race_code', 'crime_code', 'charge_degree_code']
    #y_val_columns = ['two_year_recid']

    if not os.path.exists(file_path_raw):
        df_combined.to_csv(file_path_raw)
        print("File saved successfully.")
    else:
        print("File already exists. Not saving again.")


    data_combined_scaled = np.concatenate((X_val, y_val), axis=1)
    df_combined = pd.DataFrame(data_combined_scaled)
    if not os.path.exists(file_path_scaled):
        df_combined.to_csv(file_path_scaled)
        print("File saved successfully.")
    else:
        print("File already exists. Not saving again.")

    return x_train, X_train, y_train, X_val, y_val


class Operations:

    # For now the input data is passed as init parameters
    def __init__(self, data, net):

        # Create a list of linear layers based on layer_sizes
        self.itrain = data[0]
        self.otrain = data[1]
        self.ival = data[2]
        self.oval = data[3]
        self.itrain_raw = data[4]
        self.model = net
    


    def obj_fun(self, params, minibatch):
        x = self.itrain
        y = self.otrain
        model = self.model
        #x_white = x[:, ]
        samples = np.random.choice(len(y), minibatch, replace=False)

        #print("Shapey:", x[samples, :].shape, y[samples].shape)

        fval = model.get_obj(x[samples, :], y[samples], params)
        
        #print("obj val: ", fval)

        return fval

    def obj_grad(self, params, minibatch):
        fgrad = []
        x = self.itrain
        y = self.otrain
        model = self.model
        samples = np.random.choice(len(y), minibatch, replace=False)
        #fval = model.get_obj(x[samples, :], y[samples], params)
        fgrad = model.get_obj_grad(x[samples, :], y[samples], params)

        return fgrad
    

    ## White - Black
    def conf1(self, params, minibatch):
            #print("Reached at function constraint")
            conf_val = None
            x_train = self.itrain
            y_train = self.otrain
            x_train_raw = self.itrain_raw
            model = self.model
            x_white = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0), :]
            y_white = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0)]
            x_black = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2), :]
            y_black = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2)]
            
            white_samples = np.random.choice(len(y_white), minibatch, replace=False)

            black_samples = np.random.choice(len(y_black), minibatch, replace=False)
    

            conf1 = model.get_constraint(x_white[white_samples, :], y_train[white_samples], x_black[black_samples, :], y_train[black_samples], params)
    
            #print("conf1 val: ", conf1)
            return conf1
        
    def conJ1(self, params, minibatch):
        #print("Reached at function constraint grad")
        cgrad = []
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        model = self.model
        x_white = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0), :]
        y_white = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0)]
        x_black = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2), :]
        y_black = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2)]

        white_samples = np.random.choice(len(y_white), minibatch, replace=False)
        black_samples = np.random.choice(len(y_black), minibatch, replace=False)

        conj1 = model.get_constraint_grad(x_white[white_samples, :], y_train[white_samples], x_black[black_samples, :], y_train[black_samples], params)
        #conj1 = model.get_grads(conf1, params)
        
        return conj1

    ## White - Asian
    def conf2(self, params, minibatch):
            #print("Reached at function constraint")
            conf_val = None
            x_train = self.itrain
            y_train = self.otrain
            x_train_raw = self.itrain_raw
            model = self.model
            x_white = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0), :]
            y_white = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0)]
            x_asian = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1), :]
            y_asian = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1)]
            
            white_samples = np.random.choice(len(y_white), minibatch, replace=False)

            asian_samples = np.random.choice(len(y_asian), minibatch, replace=False)
    

            conf2 = model.get_constraint(x_white[white_samples, :], y_train[white_samples], x_asian[asian_samples, :], y_train[asian_samples], params)
            
            #print("conf2 val:", conf2)

            return conf2
        
    def conJ2(self, params, minibatch):
        #print("Reached at function constraint grad")
        cgrad = []
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        model = self.model
        x_white = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0), :]
        y_white = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0)]
        x_asian = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1), :]
        y_asian = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1)]

        white_samples = np.random.choice(len(y_white), minibatch, replace=False)
        asian_samples = np.random.choice(len(y_asian), minibatch, replace=False)

        conj2 = model.get_constraint_grad(x_white[white_samples, :], y_train[white_samples], x_asian[asian_samples, :], y_train[asian_samples], params)
        #conj1 = model.get_grads(conf1, params)
        
        return conj2


    ## Black - Asian
    def conf3(self, params, minibatch):
            #print("Reached at function constraint")
            conf_val = None
            x_train = self.itrain
            y_train = self.otrain
            x_train_raw = self.itrain_raw
            model = self.model
            x_black = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2), :]
            y_black = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2)]
            x_asian = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1), :]
            y_asian = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1)]
            
            black_samples = np.random.choice(len(y_black), minibatch, replace=False)

            asian_samples = np.random.choice(len(y_asian), minibatch, replace=False)
    

            conf3 = model.get_constraint(x_black[black_samples, :], y_train[black_samples], x_asian[asian_samples, :], y_train[asian_samples], params)
    
            
            return conf3
        
    def conJ3(self, params, minibatch):
        #print("Reached at function constraint grad")
        cgrad = []
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        model = self.model
        x_black = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2), :]
        y_black = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2)]
        x_asian = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1), :]
        y_asian = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1)]

        black_samples = np.random.choice(len(y_black), minibatch, replace=False)
        asian_samples = np.random.choice(len(y_asian), minibatch, replace=False)

        conj3 = model.get_constraint_grad(x_black[black_samples, :], y_train[black_samples], x_asian[asian_samples, :], y_train[asian_samples], params)
        #conj1 = model.get_grads(conf1, params)
        
        return conj3

    ## Black - White
    def conf1_n(self, params, minibatch):
        #print("Reached at function constraint")
        conf_val = None
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        model = self.model
        x_white = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0), :]
        y_white = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0)]
        x_black = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2), :]
        y_black = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2)]
        
        white_samples = np.random.choice(len(y_white), minibatch, replace=False)

        black_samples = np.random.choice(len(y_black), minibatch, replace=False)


        conf1_n = model.get_constraint(x_black[black_samples, :], y_train[black_samples], x_white[white_samples, :], y_train[white_samples], params)
        
        return conf1_n

    def conJ1_n(self, params, minibatch):
        #print("Reached at function constraint grad")
        model = self.model
        cgrad = []
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        x_white = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0), :]
        y_white = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0)]
        x_black = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2), :]
        y_black = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2)]
        
        white_samples = np.random.choice(len(y_white), minibatch, replace=False)
        
        black_samples = np.random.choice(len(y_black), minibatch, replace=False)

        conj2 = model.get_constraint_grad(x_black[black_samples, :], y_train[black_samples], x_white[white_samples, :], y_train[white_samples], params)
        #conj2 = model.get_grads(conf2, params)

        
        return conj2

    ## Asian - White
    def conf2_n(self, params, minibatch):
        #print("Reached at function constraint")
        conf_val = None
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        model = self.model
        x_white = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0), :]
        y_white = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0)]
        x_asian = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1), :]
        y_asian = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1)]
        
        white_samples = np.random.choice(len(y_white), minibatch, replace=False)

        asian_samples = np.random.choice(len(y_asian), minibatch, replace=False)


        conf2_n = model.get_constraint(x_asian[asian_samples, :], y_train[asian_samples], x_white[white_samples, :], y_train[white_samples], params)
        
        #print("conf2_n val:",conf2_n)

        return conf2_n

    def conJ2_n(self, params, minibatch):
        #print("Reached at function constraint grad")
        model = self.model
        cgrad = []
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        x_white = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0), :]
        y_white = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0)]
        x_asian = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1), :]
        y_asian = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1)]
        
        white_samples = np.random.choice(len(y_white), minibatch, replace=False)
        
        asian_samples = np.random.choice(len(y_asian), minibatch, replace=False)

        conj2_n = model.get_constraint_grad(x_asian[asian_samples, :], y_train[asian_samples], x_white[white_samples, :], y_train[white_samples], params)
        #conj2 = model.get_grads(conf2, params)

        
        return conj2_n


    ## Asian - Black
    def conf3_n(self, params, minibatch):
        #print("Reached at function constraint")
        conf_val = None
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        model = self.model
        x_black = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2), :]
        y_black = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2)]
        x_asian = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1), :]
        y_asian = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1)]
        
        black_samples = np.random.choice(len(y_black), minibatch, replace=False)

        asian_samples = np.random.choice(len(y_asian), minibatch, replace=False)


        conf3_n = model.get_constraint(x_asian[asian_samples, :], y_train[asian_samples], x_black[black_samples, :], y_train[black_samples], params)
        
        return conf3_n

    def conJ3_n(self, params, minibatch):
        #print("Reached at function constraint grad")
        model = self.model
        cgrad = []
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        x_black = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2), :]
        y_black = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_2)]
        x_asian = x_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1), :]
        y_asian = y_train[(x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1)]
        
        black_samples = np.random.choice(len(y_black), minibatch, replace=False)
        
        asian_samples = np.random.choice(len(y_asian), minibatch, replace=False)

        conj3_n = model.get_constraint_grad(x_asian[asian_samples, :], y_train[asian_samples], x_black[black_samples, :], y_train[black_samples], params)
        #conj2 = model.get_grads(conf2, params)

        
        return conj3_n


def paramvals(maxiter, beta, rho, lamb, hess, tau, mbsz, numcon, geomp, stepdecay, gammazero, zeta, N, n, lossbound, scalef):
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
        # strategy for step decrease, options include 'dimin' 'stepwise' 'slowdimin' 'constant'
        'stepdecay': stepdecay,
        'gammazero': gammazero,  # initial stepsize
        'zeta': zeta,  # parameter associated with the stepsize iteration
        'N': N,  # Train/val sample size
        'n': n,  # Total number of parameters
        'lossbound': lossbound, #Bound on constraint loss
        'scalef': scalef #Scaling factor for constraints
    }
    return params


if __name__ == "__main__":
     
    ######Training loop######
    x_train, X_train, y_train, X_val, y_val = preprocess_data()

    parser = argparse.ArgumentParser(description="Dynamically import the model class")

    # Add argument for module name
    parser.add_argument("--model", type=str, help="Name of the model to import (backend_connect)")

    # Add argument for Optimizer name
    parser.add_argument("--optimizer", type=str, help="Optimizer Name (Default StochasticGhost)")

    # Parse command-line arguments
    args = parser.parse_args()
    model_name = args.model
    optimizer_name = args.optimizer
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(parent_dir, "../humancompatible/train")))
    #train_dir = os.path.abspath(os.path.join(parent_dir, "..", "train"))
    #sys.path.append(train_dir)
    # Dynamically import the specified module
    if model_name:
        model = importlib.import_module(model_name)

        # Get the specified function from the imported module
        CustomNetwork = getattr(model, "CustomNetwork")
    else:
        # Import the default module if no module name is provided
        print("Please specify the model architecture")

    # Import optimizer module
    if optimizer_name:
        optimizer = importlib.import_module(optimizer_name)
    else:
        # Default to StochasticGhost optimizer
        optimizer = importlib.import_module("StochasticGhost")

    print(f"Using model: {model_name}")
    print(f"Using optimizer: {optimizer_name if optimizer_name else 'StochasticGhost'}")

    loss_bound=2e-3
    trials = 21
    maxiter = 200
    acc_arr = []
    max_acc = 0
    #ftrial, ctrial1, ctrial2, ctrial3, ctrial4, ctrial5, ctrial6 = [], [], [], [], [], [], []
    ftrial, ctrial1, ctrial2, ctrial3, ctrial4 = [], [], [], [], []
    initsaved = []
    #x_train, x_val, y_train, y_val = train_test_split(in_df.values, out_df.values, test_size=0.3, random_state=42)
    ip_size = x_train.shape[1]
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # Y_train = torch.tensor(y_train, dtype=torch.float32)
    # X_val = torch.tensor(X_val, dtype=torch.float32)
    # Y_val = torch.tensor(y_val, dtype=torch.float32)
    saved_model = []

    for trial in range(trials):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>TRIAL", trial)
        
        #print(type(X_train))
        hid_size1 = 16
        hid_size2 = 16
        op_size = 1
        layer_sizes = [ip_size, hid_size1, hid_size2, op_size]
        data = (X_train[:, :ip_size], y_train, X_val[:, :ip_size], y_val, x_train)
        model_specs = (layer_sizes,)
        #x_len = x_train[:, 4]
        num_trials = min(len(y_train[((x_train[:, RACE_IND]) == SENSITIVE_CODE_1)]), len(y_train[(x_train[:, RACE_IND] == SENSITIVE_CODE_0)]), len(y_train[((x_train[:, RACE_IND]) == SENSITIVE_CODE_2)]))


        print(num_trials)
        #print(num_trials)
        net = CustomNetwork(model_specs)
        operations = Operations(data, net)
        
        initw, num_param = net.get_trainable_params()
        #cons_list = [operations.conf1, operations.conf2, operations.conf3, operations.conf1_n, operations.conf2_n, operations.conf3_n]
        #cons_grad_list = [operations.conJ1, operations.conJ2, operations.conJ3, operations.conJ1_n, operations.conJ2_n, operations.conJ3_n]
        cons_list = [operations.conf1, operations.conf2, operations.conf1_n, operations.conf2_n]
        cons_grad_list = [operations.conJ1, operations.conJ2, operations.conJ1_n, operations.conJ2_n]
        #scalef_list = [1., 1., 1., 1., 1., 1.]
        scalef_list = [1., 1., 1., 1.]
        #loss_bound_list = [loss_bound, loss_bound, loss_bound, loss_bound, loss_bound, loss_bound]
        loss_bound_list = [loss_bound, loss_bound, loss_bound, loss_bound]
        # print(len(initw))
        

        """ Add a conditional block for your optimizer
            In general, this module expects the optimizer to return the learned parameters (w in this case)
            iterfs and itercs are just for plotting purposes
            the function args could be: objective and constraint function, objective and constraint grads, sampling parameter etc
            
            StochasticGhost Optimizer takes:
            operations.obj_fun : objective function definition
            operations.obj_grad : objective function grads
            list constraints : constraint functions (list)
            list constraint grads : constraint function gradients (list)
            initw : learnable parameters
            params : the dictionary of hyperparameters (details described in StocgasticGhost module)
        """
        if optimizer_name == "StochasticGhost":
            params = paramvals(maxiter=maxiter, beta=10., rho=1e-3, lamb=0.5, hess='diag', tau=16., mbsz=100,
                            numcon=4, geomp=0.2, stepdecay='dimin', gammazero=0.1, zeta=0.4, N=num_trials, n=num_param, lossbound=loss_bound_list, scalef=scalef_list)
            w, iterfs, itercs = optimizer.StochasticGhost(operations.obj_fun, operations.obj_grad, cons_list, cons_grad_list, initw, params)
        
        if np.isnan(w[0]).any():
            print("reached infeasibility not saving the model")
        else:
            ftrial.append(iterfs)
            ctrial1.append(itercs[:,0])
            ctrial2.append(itercs[:,1])
            ctrial3.append(itercs[:,2])
            ctrial4.append(itercs[:,3])
            #ctrial5.append(itercs[:,4])
            #ctrial6.append(itercs[:,5])

            saved_model.append(net)
            #torch.save(net, 'coverage_models_tr/saved_model'+str(trial)+'.pth')
            directory = "../saved_models/"+str(model_name)

            # Check if the directory exists
            if not os.path.exists(directory):
                # If the directory doesn't exist, create it
                os.makedirs(directory)

            # Save the model
            model_path = os.path.join(directory, f'saved_model_coverage{trial}')
            net.save_model(model_path)
            print(f'model_{trial} saved')

    ftrial = np.array(ftrial).T
    ctrial1 = np.array(ctrial1).T
    ctrial2 = np.array(ctrial2).T
    ctrial3 = np.array(ctrial3).T
    ctrial4 = np.array(ctrial4).T
    #ctrial5 = np.array(ctrial5).T
    #ctrial6 = np.array(ctrial6).T
    print(">>>>>>>>>>>>>>>>>>>Completed trials<<<<<<<<<<<<<<<<")
    #print(acc_arr)
    df_ftrial = pd.DataFrame(ftrial, columns=range(1, ftrial.shape[1]+1), index=range(1, ftrial.shape[0]+1))
    df_ctrial1 = pd.DataFrame(ctrial1, columns=range(1, ctrial1.shape[1]+1), index=range(1, ctrial1.shape[0]+1))
    df_ctrial2 = pd.DataFrame(ctrial2, columns=range(1, ctrial2.shape[1]+1), index=range(1, ctrial2.shape[0]+1))
    df_ctrial3 = pd.DataFrame(ctrial3, columns=range(1, ctrial3.shape[1]+1), index=range(1, ctrial3.shape[0]+1))
    df_ctrial4 = pd.DataFrame(ctrial4, columns=range(1, ctrial4.shape[1]+1), index=range(1, ctrial4.shape[0]+1))
    #df_ctrial5 = pd.DataFrame(ctrial5, columns=range(1, ctrial5.shape[1]+1), index=range(1, ctrial5.shape[0]+1))
    #df_ctrial6 = pd.DataFrame(ctrial6, columns=range(1, ctrial6.shape[1]+1), index=range(1, ctrial6.shape[0]+1))

    # Save DataFrames to CSV files
    df_ftrial.to_csv('../utils/coverage_ftrial_'+str(loss_bound)+'.csv')
    df_ctrial1.to_csv('../utils/coverage_ctrial1_'+str(loss_bound)+'.csv')
    df_ctrial2.to_csv('../utils/coverage_ctrial2_'+str(loss_bound)+'.csv')
    df_ctrial3.to_csv('../utils/coverage_ctrial2_'+str(loss_bound)+'.csv')
    df_ctrial4.to_csv('../utils/coverage_ctrial2_'+str(loss_bound)+'.csv')
    #df_ctrial5.to_csv('../utils/coverage_ctrial2_'+str(loss_bound)+'.csv')
    #df_ctrial6.to_csv('../utils/coverage_ctrial2_'+str(loss_bound)+'.csv')

