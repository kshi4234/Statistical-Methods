import kagglehub
import os

os.environ['KAGGLEHUB_CACHE'] = './data'
path = kagglehub.dataset_download("suddharshan/retail-price-optimization")

print(f'Path to dataset: {path}')