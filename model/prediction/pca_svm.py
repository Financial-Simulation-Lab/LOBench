import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from config import Config
import data 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
import torch

def create_dataloaders(c):
    dataset_params = c.get_dict('data.dataset')
    dataloader_params = c.get_dict('data.dataloader')
    dataset_class = getattr(data, dataset_params['name'])
    dataloader_class = getattr(data, dataloader_params['name'])
         
    ds = dataset_class(**dataset_params)
    dl = dataloader_class(datasets=ds, **dataloader_params)
      
    train_dataloader = dl.train_dataloader()
    val_dataloader = dl.val_dataloader()
    test_dataloader = dl.test_dataloader()
        
    return train_dataloader, val_dataloader, test_dataloader

def dataloader_to_numpy(dataloader):
    X_list, y_list = [], []
    for batch in dataloader:
        X_batch, y_batch = batch
        X_list.append(X_batch.numpy()) 
        y_list.append(y_batch.numpy())
    X = np.vstack(X_list) 
    y = np.hstack(y_list) 
    return X, y

if __name__ == "__main__":
    c = Config("experiment/exp_settings/default.json")
    c.update_config("experiment/exp_settings/pred_FI_2010/svm.json")
    
    train_dl, val_dl, test_dl=create_dataloaders(c)
    X_train,y_train = dataloader_to_numpy(train_dl)
    X_val,y_val = dataloader_to_numpy(val_dl)
    X_test,y_test = dataloader_to_numpy(test_dl)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    
    n_components = 128
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_reshaped)
    X_test_pca = pca.transform(X_test_reshaped)
    
    param_grid = {
        'C': [0.1, 1, 10],          
        'kernel': ['linear', 'rbf'],     
        'gamma': [0.01, 0.1, 1], 
    }

    svm = SVC(probability=True,verbose=1)
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train_pca,y_train)
    print("Best Parameters from Grid Search:", grid_search.best_params_)
    with open('svm_model.pkl', 'wb') as file:
        pickle.dump(svm, file)
        print("Model saved successfully!")
        
    