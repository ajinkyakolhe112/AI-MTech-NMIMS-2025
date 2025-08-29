import datasets as huggingface_datasets

training_dataset   = huggingface_datasets.load_dataset("mnist", split="train")

import pandas as pd
training_dataframe = pd.read_csv("datasets/mnist_train.csv")

# TODO: Create Pytorch Dataset from the variable training_dataset
# TODO: Create a DataLoader from the Pytorch Dataset
# TODO: Create a Pytorch Model
# TODO: Create Pytorch Training Loop (with Pytorch Lightning)
# TODO: Model Prediction

class Custom_Dataset_MNIST(torch.utils.data.Dataset):
    def __init__(self):
        training_dataframe = pd.read_csv("datasets/mnist_train.csv")
        y_actual = training_dataframe['label'].values()
        x_actual = training_dataframe.iloc[:,1:]
        
    def __getitem(self, index):
        pass
        
    def __len__(self):
        pass

dataset_variable =  Custom_Dataset_MNIST() 
