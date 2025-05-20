import numpy as np
import pandas as pd
import os

print(os.getcwd())

def load_data(filename):
    df = pd.read_csv(filename)
    return df

filename = "E:\...DATA SCIENCE AND QUANT RESOURCES\Classical-Statistical-Methods\Projects\Time Series\Taxes_and_Economy\data\GDP_Growth.csv"
df = load_data(filename)