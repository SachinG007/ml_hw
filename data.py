import os
import gzip
import urllib
# import urllib.requests
import numpy as np
import tarfile
import pickle 
import torch

CIFAR10_FOLDER = 'cifar-10-batches-py'
SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_DIRECTORY = "./data/"

def download_cifar_data():
    """Downloads the CIFAR-10 data (if it hasnt been downloaded yet), and returns the path to the data directory."""
    
    if not os.path.exists(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, "cifar-10-python.tar.gz")
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL, filepath)
        size = os.stat(filepath).st_size
        print('Successfully downloaded CIFAR data ', size, 'bytes.')
        # Extract and prepare CIFAR10 DATA
        print("Extracting Python CIFAR10 data.")
        tarfile.open(filepath, 'r:gz').extractall(DATA_DIRECTORY)
        
    return DATA_DIRECTORY

def load_cifar_data(data_dir):
    """Takes as input the path to the CIFAR data directory, loads the data and returns (train_data, train_labels, test_data, test_labels)."""
    
    if CIFAR10_FOLDER in os.listdir(data_dir) and  'data_batch_1' not in os.listdir(data_dir):
        # move the data directory down one
        data_dir = os.path.join(data_dir, CIFAR10_FOLDER)
    train_files = ['data_batch_'+str(x) for x in range(1,6)]
    train_files = [os.path.join(data_dir, f) for f in train_files]
    test_files = ['test_batch']
    test_files = [os.path.join(data_dir, f) for f in test_files]
    num_classes = 10
    label_func = lambda x: np.array(x['labels'], dtype='int32')
    
    # Load the data into memory
    def load_files(filenames):
        data = np.array([])
        labels = np.array([])
        for name in filenames:
            with open(name, 'rb') as f:
                mydict = pickle.load(f, encoding='latin1')

            # The labels have different names in the two datasets.
            newlabels = label_func(mydict)
            if data.size:
                data = np.vstack([data, mydict['data']])
                labels = np.hstack([labels, newlabels])
            else:
                data = mydict['data']
                labels = newlabels
        data = np.reshape(data, [-1, 3, 32, 32], order='C')
        data = np.transpose(data, [0, 2, 3, 1])
        data = data/255.
        return data, labels

    train_data, train_labels = load_files(train_files)
    test_data, test_labels = load_files(test_files)
    
    return train_data, train_labels, test_data, test_labels

def prepare_cifar_data(n_train=10000, n_val=10000, n_test=10000):
    data_dir = download_cifar_data()
    X_train_full, y_train_full, X_test, y_test = load_cifar_data(data_dir)
    
    
    X_train = X_train_full[:n_train]
    X_val = X_train_full[n_train:n_train+n_val]
    X_test = X_test
    y_train = y_train_full[:n_train]
    y_val = y_train_full[n_train:n_train+n_val]
    y_test = y_test


    assert len(X_train)==len(y_train)
    assert len(X_val)==len(y_val)
    assert len(X_test)==len(y_test)

    data = {'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test}
    return data

def get_dataloader(X, y, batch_size):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader