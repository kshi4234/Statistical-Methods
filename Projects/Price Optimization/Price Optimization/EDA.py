import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def check_correlation(df: pd.DataFrame, title: str, col1: str, col2: str):
    fig, ax = plt.subplots()
    ax.scatter(df[col1], df[col2])
    ax.set_title(f'{title}: {col2} vs {col1}')
    ax.set_xlabel(f'{col1}')
    ax.set_ylabel(f'{col2}')
    plt.show()
    
# Given the product_id, qty and total_price are highly correlated
def explore(df: pd.DataFrame):
    print(df.head(5))
    print(df.columns)
    
    groups = df.groupby(by=['product_id'])
    for group_key, group in groups:
        check_correlation(group, group_key, 'qty', 'total_price')

if __name__ == '__main__':
    df = pd.read_csv('D:/..PERSONAL PROJECTS/Pricing Simulation/data/datasets/suddharshan/retail-price-optimization/versions/2/retail_price.csv')
    explore(df)