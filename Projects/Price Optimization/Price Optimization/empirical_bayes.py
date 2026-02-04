import pandas as pd
import numpy as np
from utils import plot_model

# This program operates under the assumption of normally distributed residuals and a normally distributed prior.
# Further, implicitly assume that sigma = sigma^2 and alpha = alpha^2 for simplicity's sake.

def implicit_alpha(design_mat, mean_vec, sigma, alpha):
    """
    implicit_alpha: This method describes the formula for the optimal prior variance parameter
    
    Returns:
        n_alpha - The new estimate for the parameter
        gamma - A number computed from eigenvalues
    """
    coeff = (1.0 / sigma) * np.transpose(design_mat) @ design_mat
    eig_vals, eig_vecs = np.linalg.eigh(coeff)
    # print(sorted(eig_vals))
    # quit()
    gamma = 0
    
    for eig_val in eig_vals:
        gamma += eig_val / ((1.0 / alpha) + eig_val)
    
    n_alpha = np.dot(mean_vec, mean_vec) / gamma
    return n_alpha, gamma

def implicit_sigma(design_mat, mean_vec, y, gamma):
    """
    implicit_sigma: This method describes the formula for the optimal residual variance parameter
    """
    n = len(design_mat)
    coeff = (1.0 / (n - gamma))
    vec = (y - design_mat @ mean_vec)
    n_sigma = coeff * np.dot(vec, vec)
    return n_sigma

def design_matrix(price, product_ids: dict):
    category_prods = [product_ids[key] for key in product_ids.keys()]
    category_prods = category_prods[:-1]
    
    interactions = [price * category for category in category_prods]
    
    dm = np.concatenate((interactions, category_prods, [[1 for _ in range(len(price))]]))
    dm = np.transpose(dm)
    dm += 1e-7
    # print(dm[0][:])
    # quit()
    return dm

def mean_vector(precision_mat, design_mat, sigma, y):
    covariance = np.linalg.inv(precision_mat)
    # print('--------------')
    # print(covariance)
    mu = (1.0 / sigma) * covariance @ (np.transpose(design_mat) @ y)
    # print(mu)
    return mu

def precision_matrix(design_mat, sigma, alpha):
    m = len(design_mat[0]) 
    precision = (1.0 / sigma) * (np.transpose(design_mat) @ design_mat) + (1.0 / alpha) * np.eye(m)
    return precision

def make_predictions(design_mat, posterior_mean, posterior_variance, sigma):
    n = len(design_mat)
    predictive_mean = design_mat @ posterior_mean
    predictive_variance = sigma * np.eye(n) + design_mat @ posterior_variance @ np.transpose(design_mat)
    predictions = np.random.multivariate_normal(mean=predictive_mean, cov=predictive_variance)
    print(max(predictions))
    predictions = np.minimum(predictions, 130)
    return predictions

def main():
    df = pd.read_csv('D:/..PERSONAL PROJECTS/Price Optimization/data/datasets/suddharshan/retail-price-optimization/versions/2/retail_price.csv')
    df['orig_product_id'] = df['product_id']
    df = pd.get_dummies(df, prefix='d', columns=['product_id'])
    
    print(max(df['qty']))
    
    one_hots = df.filter(regex='^d_')
    
    product_ids = one_hots.to_dict(orient='list')
    
    # print(one_hots.head())
    # exit()
    
    dm = design_matrix(df['total_price'].to_numpy(), product_ids)
    dm = (dm - dm.mean(axis=0)) / dm.std(axis=0)
    # print(dm)
    y = df['qty'].to_numpy()
    alpha, sigma = 1.0, 1.0 # Initial estimate for parameters
    prev_alpha, prev_sigma = 0.0, 0.0
    epsilon = 1e-10  # Threshold for convergence to a fixed point
    while np.abs(alpha-prev_alpha) > epsilon or np.abs(sigma-prev_sigma) > epsilon:
        prev_alpha, prev_sigma = alpha, sigma
        pm = precision_matrix(dm, sigma, alpha)
        mu = mean_vector(pm, dm, sigma, y)
        temp_alpha, gamma = implicit_alpha(dm, mu, sigma, alpha)
        temp_sigma = implicit_sigma(dm, mu, y, gamma)
        alpha, sigma = temp_alpha, temp_sigma   # Update the parameters together
        print(f'alpha: {alpha}, sigma: {sigma}')    # Debugging 
    
    posterior_precision = precision_matrix(dm, sigma, alpha)
    posterior_variance = np.linalg.inv(posterior_precision)
    posterior_mean = mean_vector(posterior_precision, dm, sigma, y)
    
    # print(posterior_mean)
    
    preds = make_predictions(dm, posterior_mean, posterior_variance, sigma)
    # print(preds)
    df['pred_qty'] = preds
    plot_model(df, preds, 'price', 'qty', type='pred')
    
if __name__ == '__main__':
    main()