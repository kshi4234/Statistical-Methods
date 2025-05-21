import numpy as np
import pandas as pd
import os

def load_data(filename, header=0, type='csv', encoding='utf-8'):
    df = ''
    if type == 'csv':
        df = pd.read_csv(filename, header=header, encoding=encoding)
    return df

def load_dataset():
    # Gross Domestic Product
    gdp_file = "E:/...DATA SCIENCE AND QUANT RESOURCES/Classical-Statistical-Methods/Projects/Time Series/Taxes_and_Economy/data/GDP_Growth.csv"
    # National Health Expenditures   
    nhe_file = "E:/...DATA SCIENCE AND QUANT RESOURCES/Classical-Statistical-Methods/Projects/Time Series/Taxes_and_Economy/data/NHE2023.csv"
    # Military Expenditures  
    milex_file = "E:/...DATA SCIENCE AND QUANT RESOURCES/Classical-Statistical-Methods/Projects/Time Series/Taxes_and_Economy/data/API_MS.MIL.XPND.CD_DS2_en_csv_v2_85458.csv"  
    # Tax files
    tax_file = "E:/...DATA SCIENCE AND QUANT RESOURCES/Classical-Statistical-Methods/Projects/Time Series/Taxes_and_Economy/data/Historical Income Tax Rates and Brackets, 1862-2021.csv" 
    
    variables = ['explanatory variables:', 'nhe_data','milex_data','tax_data']
    variable_files = ['explanatory variables:', (nhe_file, 1), (milex_file, 2), (tax_file, 0)]
    target = ['response/target:', 'gdp_data']

    dataset = {}
    
    # Pandas dataframe objects
    df_gdp = load_data(gdp_file, header=2, encoding="unicode_escape")
    dataset[target[1]] = df_gdp
    
    for i in range(1, len(variables)):
        # print(variables[i])
        dataset[variables[i]] = load_data(variable_files[i][0], header=variable_files[i][1], encoding="unicode_escape")
    return dataset, variables, target
    