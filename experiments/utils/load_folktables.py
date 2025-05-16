import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append("..")

from folktables import ACSDataSource, ACSPublicCoverage, ACSEmployment, ACSIncome, generate_categories

RAC1P_WHITE = 1

def load_folktables_torch(dataset: str = 'employment', state='AL', random_state=None, onehot=True, make_unbalanced=False):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'raw_data', dataset))
    data_source = ACSDataSource(survey_year=2018, horizon='1-Year', survey='person', root_dir=data_dir)
    definition_df = data_source.get_definitions(download=True)
    acs_data = data_source.get_data(states=[state], download=True)
    # group here refers to race (RAC1P)
    if dataset == 'employment':
        features, label, group = ACSEmployment.df_to_numpy(acs_data)
        # drop the RAC1P column
        features = features[:,:-1]
    elif dataset == 'coverage':
        features, label, group = ACSPublicCoverage.df_to_numpy(acs_data)
    elif dataset == 'income':
        if onehot:
            categories = generate_categories(features=ACSIncome.features, definition_df=definition_df)
            features, label, group = ACSIncome.df_to_pandas(acs_data, categories=categories, dummies=True)
            sens_features = [col for col in features.columns if col.startswith('RAC1P')]
            features = features.drop(columns=sens_features).to_numpy(dtype='float')
            label = label.to_numpy(dtype='float')
        else:
            features, label, group = ACSIncome.df_to_numpy(acs_data)
            # drop the RAC1P column
            features = features[:,:-1]
    
    # drop sensitive
    group_binary = (group == RAC1P_WHITE).astype(float)
        
    # stratify by binary race (white vs rest)
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        features, label, group_binary, test_size=0.2, stratify = group_binary, random_state=random_state)
    
    if make_unbalanced:
        # g_train_new = g_train[:len(g_train)/2]
        train_w_idx = np.argwhere(g_train == 1).flatten()
        train_nw_idx = np.argwhere(g_train != 1).flatten()
        train_nw_idx = train_nw_idx[:len(train_nw_idx)//10]
        idx = np.concatenate([train_w_idx, train_nw_idx])
        X_train = X_train[idx]
        y_train = y_train[idx]
        g_train = g_train[idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_w_idx = np.argwhere(g_train == 1).flatten()
    train_nw_idx = np.argwhere(g_train != 1).flatten()
    
    test_w_idx = np.argwhere(g_test == 1).flatten()
    test_nw_idx = np.argwhere(g_test != 1).flatten()
    
    # pdX_train_scaled
    
    return X_train_scaled, y_train, [train_w_idx, train_nw_idx], X_test_scaled, y_test, [test_w_idx, test_nw_idx]