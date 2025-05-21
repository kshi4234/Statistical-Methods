import numpy as np
from utils import data_loading

# Impute missing values in the dataset using regression, as there is not a lot of it.
def impute(dataset, variables, target):
    return

def clean(dataset, variables, target):
    """ Some notes on what to clean.
    1) Column 2024. GDP growth data only goes to 2023, not sure why it's included
    2) Call impute to impute the missing values 
    """
    return

# Do some exploratory analysis of the data. 
def data_exploration(dataset, variables, target):
    """
    For my purposes, I will simply plot the United States data as a function of time.
    This will serve to help me decide what types of modeling would be suited for my
    dataset.
    
    Tax data is quite complicated, as oftentimes the tax rates change, to say nothing of
    how the number of brackets change as well. 
    """
    gdp_data = dataset[target[1]]
    health_data = dataset[variables[1]]
    milex_data = dataset[variables[2]]
    # TODO: Tax data I want to separate into 2 different variables, where they will 
    # TODO: be split into single filer tax brackets. One will be the highest tax bracket,
    # TODO: the other will be the lowest.
    tax_data = dataset[variables[3]]
    us_gdp = gdp_data[gdp_data['Country Code']=='USA']
    us_health = health_data[health_data['Expenditure Amount (Millions)']=='Total National Health Expenditures']
    us_milex = milex_data[milex_data['Country Code']=='USA']
    
    print(us_gdp)
    print(us_health)
    print(us_milex)
    

def main():
    dataset, variables, target = data_loading.load_dataset()
    print(variables)
    print(target)
    print(dataset[variables[1]])
    data_exploration(dataset, variables, target)
    
if __name__ == '__main__':
    main()