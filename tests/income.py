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
from humancompatible.train.stochastic_ghost import StochasticGhost
import argparse

sys.path.append("..")  # Add parent directory to the sys.path



GENDER_IND = 2
SENSITIVE_CODE_0 = 0
SENSITIVE_CODE_1 = 1


def preprocess_data():
    #adult = pd.read_csv('adult.csv')

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    # File path of the data file
    file_path = os.path.join(data_dir, 'adult.data')
    adult = pd.read_csv(file_path, names=column_names)

    ############# If you want to drop the missing value rows ################
    mask = adult.eq(' ?')
    mask.sum()
    adult = adult.drop(adult.index[mask.any(axis=1)]).reset_index(drop=True)
    len(adult)

    # Setting all the categorical columns to type category
    for col in set(adult.columns) - set(adult.describe().columns):
        adult[col] = adult[col].astype('category')


    adult['hour_worked_bins'] = ['<40' if i < 40 else '40-60' if i <= 60 else '>60'  for i in adult['hours-per-week']]
    adult['hour_worked_bins'] = adult['hour_worked_bins'].astype('category')

    # Create one hot encoding of the categorical columns in the data frame.
    def oneHotCatVars(df, df_cols):
        
        df_1 = adult_data = df.drop(columns = df_cols, axis = 1)
        df_2 = pd.get_dummies(df[df_cols])
        
        return (pd.concat([df_1, df_2], axis=1, join='inner'))


    gender_mapping = {' Male': SENSITIVE_CODE_0, ' Female': SENSITIVE_CODE_1}

    adult_new = adult.copy()
    print("Unique values in 'gender' column in the new DataFrame:", adult_new['gender'].unique())

    # Create a new column 'race_code' based on the mapping
    adult_new['gender_code'] = adult_new['gender'].map(gender_mapping)

    income_mapping = {' >50K': 1, ' <=50K': 0}
    adult_new['income_code'] = adult_new['income'].map(income_mapping)

    adult_new['education_code'] = pd.Categorical(adult_new['education']).codes
    adult_new['marital_code'] = pd.Categorical(adult_new['marital-status']).codes
    adult_new['occupation_code'] = pd.Categorical(adult_new['occupation']).codes
    adult_new['relationship_code'] = pd.Categorical(adult_new['relationship']).codes
    adult_new['race_code'] = pd.Categorical(adult_new['race']).codes
    adult_new['country_code'] = pd.Categorical(adult_new['native-country']).codes
    adult_new['hours_code'] = pd.Categorical(adult_new['hour_worked_bins']).codes

    in_cols = ['education_code', 'marital_code', 'gender_code', 'occupation_code', 'relationship_code', 'race_code', 'country_code', 'hours_code']
    out_cols = ['income_code']


    x_train, x_val, y_train, y_val = train_test_split(adult_new[in_cols].values, adult_new[out_cols].values, test_size  = 0.30)

    # Normalization

    scaler = StandardScaler()  

    # Fitting only on training data
    scaler.fit(x_train)  
    X_train = scaler.transform(x_train)  

    # Applying same transformation to test data
    X_val = scaler.transform(x_val)

    # Assuming x_val and y_val are numpy arrays
    # Convert y_val to a column vector to match the shape of x_val
    #y_val = np.expand_dims(y_val, axis=1)

    # Concatenate x_val and y_val along the columns
    file_path_raw = os.path.join('../data/val_data', 'val_data_raw_income.csv')
    file_path_scaled = os.path.join('../data/val_data', 'val_data_scaled_income.csv')

    data_combined_raw = np.concatenate((x_val, y_val), axis=1)

    # Convert the combined data to a DataFrame
    df_combined = pd.DataFrame(data_combined_raw)

    #x_val_columns = ['priors_count', 'score_code', 'age_code', 'gender_code', 'race_code', 'crime_code', 'charge_degree_code']
    #y_val_columns = ['two_year_recid']

    if not os.path.exists(file_path_raw):
        df_combined.to_csv(file_path_raw, index=False)
        print("File saved successfully.")
    else:
        print("File already exists. Not saving again.")


    data_combined_scaled = np.concatenate((X_val, y_val), axis=1)
    df_combined = pd.DataFrame(data_combined_scaled)
    if not os.path.exists(file_path_scaled):
        df_combined.to_csv(file_path_scaled, index=False)
        print("File saved successfully.")
    else:
        print("File already exists. Not saving again.")

    
    return  x_train, X_train, y_train, X_val, y_val



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
        #x_female = x[:, ]
        samples = np.random.choice(len(y), minibatch, replace=False)

        fval = model.get_obj(x[samples, :], y[samples], params)
        
        
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
    


    def conf1(self, params, minibatch):
        #print("Reached at function constraint")
        conf_val = None
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        model = self.model
        x_female = x_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_1), :]
        y_female = y_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_1)]
        x_male = x_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_0), :]
        y_male = y_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_0)]
        
        female_samples = np.random.choice(len(y_female), minibatch, replace=False)

        male_samples = np.random.choice(len(y_male), minibatch, replace=False)
   

        conf1 = model.get_constraint(x_female[female_samples, :], y_train[female_samples], x_male[male_samples, :], y_train[male_samples], params)
   
        
        return conf1
    
    def conJ1(self, params, minibatch):
        #print("Reached at function constraint grad")
        cgrad = []
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        model = self.model
        x_female = x_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_1), :]
        y_female = y_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_1)]
        x_male = x_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_0), :]
        y_male = y_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_0)]

        female_samples = np.random.choice(len(y_female), minibatch, replace=False)
        male_samples = np.random.choice(len(y_male), minibatch, replace=False)

        conj1 = model.get_constraint_grad(x_female[female_samples, :], y_train[female_samples], x_male[male_samples, :], y_train[male_samples], params)
        #conj1 = model.get_grads(conf1, params)
        
        return conj1

    def conf2(self, params, minibatch):
        #print("Reached at function constraint")
        conf_val = None
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        model = self.model
        x_female = x_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_1), :]
        y_female = y_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_1)]
        x_male = x_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_0), :]
        y_male = y_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_0)]
        
        female_samples = np.random.choice(len(y_female), minibatch, replace=False)

        male_samples = np.random.choice(len(y_male), minibatch, replace=False)
    

        conf2 = model.get_constraint(x_male[male_samples, :], y_train[male_samples], x_female[female_samples, :], y_train[female_samples], params)
        
        return conf2
    
    def conJ2(self, params, minibatch):
        #print("Reached at function constraint grad")
        model = self.model
        cgrad = []
        x_train = self.itrain
        y_train = self.otrain
        x_train_raw = self.itrain_raw
        x_female = x_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_1), :]
        y_female = y_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_1)]
        x_male = x_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_0), :]
        y_male = y_train[(x_train_raw[:, GENDER_IND] == SENSITIVE_CODE_0)]
        
        female_samples = np.random.choice(len(y_female), minibatch, replace=False)
        
        male_samples = np.random.choice(len(y_male), minibatch, replace=False)

        conj2 = model.get_constraint_grad(x_male[male_samples, :], y_train[male_samples], x_female[female_samples, :], y_train[female_samples], params)
        #conj2 = model.get_grads(conf2, params)

        
        return conj2
    
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

    # Parse command-line arguments
    args = parser.parse_args()
    model_name = args.model
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(parent_dir, "..")))
    # Dynamically import the specified module
    if model_name:
        model = importlib.import_module(model_name)

        # Get the specified function from the imported module
        CustomNetwork = getattr(model, "CustomNetwork")
    else:
        # Import the default module if no module name is provided
        print("Please specify the model architecture")

    # Call the function from the imported module
    loss_bound=2e-3
    trials = 21
    maxiter = 200
    acc_arr = []
    max_acc = 0
    ftrial, ctrial1, ctrial2 = [], [], []
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
        num_trials = min(len(y_train[((x_train[:, GENDER_IND]) == SENSITIVE_CODE_1)]), len(y_train[(x_train[:, GENDER_IND] == SENSITIVE_CODE_0)]))


        print(num_trials)
        #print(num_trials)
        net = CustomNetwork(model_specs)
        operations = Operations(data, net)
        
        initw, num_param = net.get_trainable_params()
        # print(len(initw))
        params = paramvals(maxiter=maxiter, beta=10., rho=1e-3, lamb=0.5, hess='diag', tau=2., mbsz=100,
                        numcon=2, geomp=0.2, stepdecay='dimin', gammazero=0.1, zeta=0.4, N=num_trials, n=num_param, lossbound=[loss_bound, loss_bound], scalef=[1., 1.])
        w, iterfs, itercs = StochasticGhost(operations.obj_fun, operations.obj_grad, [operations.conf1, operations.conf2], [operations.conJ1, operations.conJ2], initw, params)
        
        if np.isnan(w[0]).any():
            print("reached infeasibility not saving the model")
        else:
            ftrial.append(iterfs)
            ctrial1.append(itercs[:,0])
            ctrial2.append(itercs[:,1])

            saved_model.append(net)
            #torch.save(net, 'income_models_tr/saved_model'+str(trial)+'.pth')
            directory = "../saved_models/"+str(model_name)

            # Check if the directory exists
            if not os.path.exists(directory):
                # If the directory doesn't exist, create it
                os.makedirs(directory)

            # Save the model
            model_path = os.path.join(directory, f'saved_model_income{trial}')
            net.save_model(model_path)

    ftrial = np.array(ftrial).T
    ctrial1 = np.array(ctrial1).T
    ctrial2 = np.array(ctrial2).T
    print(">>>>>>>>>>>>>>>>>>>Completed trials<<<<<<<<<<<<<<<<")
    #print(acc_arr)
    df_ftrial = pd.DataFrame(ftrial, columns=range(1, ftrial.shape[1]+1), index=range(1, ftrial.shape[0]+1))
    df_ctrial1 = pd.DataFrame(ctrial1, columns=range(1, ctrial1.shape[1]+1), index=range(1, ctrial1.shape[0]+1))
    df_ctrial2 = pd.DataFrame(ctrial2, columns=range(1, ctrial2.shape[1]+1), index=range(1, ctrial2.shape[0]+1))

    # Save DataFrames to CSV files
    df_ftrial.to_csv('../utils/income_ftrial_new.csv')
    df_ctrial1.to_csv('../utils/income_ctrial1_new.csv')
    df_ctrial2.to_csv('../utils/income_ctrial2_new.csv')