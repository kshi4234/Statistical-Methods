import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    df['pred_qty'] = res.predict(df)
    if type == 'log':
        df['pred_qty'] = np.exp(df['pred_qty'])
    groups_df = df.groupby(by=['product_id'])
    for group_key, group in groups_df:
        plot(group_key, group, col1, col2)

# Elasticity model to perform modeling of point elasticity as opposed to direct
# prediction. Will use log-log regression.
def elasticity_model(df: pd.DataFrame):
    model = smf.ols(formula='log_qty ~ log_price + C(product_id) + C(product_id):log_price', data=df)
    res = model.fit()
    print(res.params)
    print(res.summary())
    return model, res

def create_model(df: pd.DataFrame):
    # Include interactions term between the id (first making it categorical) and the price
    # Predict the qty sold
    # NOTE: Capital 'C' turns categorical variable
    # NOTE: ':' multiplies two terms; '*' also does so BUT INCLUDES the 'marginal' columns
    # in other words, (x*y) <=> (x + y + x:y)
    model = smf.ols(formula='qty ~ total_price + C(product_id) + C(product_id):total_price', data=df)
    res = model.fit()
    print(res.params)
    print(res.summary())
    return model, res

if __name__ == '__main__':
    df = pd.read_csv('D:/..PERSONAL PROJECTS/Pricing Simulation/data/datasets/suddharshan/retail-price-optimization/versions/2/retail_price.csv')
    model, res = create_model(df)
    # plot_model(df, res)
    
    df['log_qty'] = np.log(df['qty'])
    df['log_price'] = np.log(df['total_price'])
    model, res = elasticity_model(df)
    plot_model(df, res, 'log_price', 'log_qty', type='log')