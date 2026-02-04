import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot(group_key: str, df: pd.DataFrame, col1: str, col2: str):
    fig, ax = plt.subplots()
    df = df.sort_values(by=['total_price'])
    ax.plot(df['total_price'], df['qty'], color='b')
    ax.plot(df['total_price'], df['pred_qty'], color='r')
    ax.set_title(label=f'{group_key}', loc='center')
    ax.set_xlabel(xlabel=f'{col1}')
    ax.set_ylabel(ylabel=f'{col2}')
    plt.show()
    
def plot_model(df: pd.DataFrame, res, col1: str, col2: str, type: str = 'pred'):
    # df['pred_qty'] = res.predict(df)
    if type == 'log':
        df['pred_qty'] = np.exp(df['pred_qty'])
    group = 'orig_product_id' if 'product_id' not in df.columns else 'product_id'
    groups_df = df.groupby(by=[group])
    for group_key, group in groups_df:
        plot(group_key, group, col1, col2)