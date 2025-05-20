import numpy as np
import pandas as pd
import os

print(os.getcwd())

def load_data(filename, header=0):
    df = pd.read_csv(filename, header=header)
    return df


gdp_file = "E:\...DATA SCIENCE AND QUANT RESOURCES\Classical-Statistical-Methods\Projects\Time Series\Taxes_and_Economy\data\GDP_Growth.csv"
df_gdp = load_data(gdp_file, header=2)  # Pandas dataframe object
print(df_gdp)