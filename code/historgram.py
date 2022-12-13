# import torch
import matplotlib
import matplotlib.pyplot as plt
from datasets import Dataset


data_path = "../data"
name = ['YAGO15K', 'ICEWS14']
for i in name:  
    dataset = Dataset(data_path, i)
    time_diff = dataset.calculate()
    plt.bar(list(time_diff.keys()), time_diff.values(), color='g')
    plt.show()
    plt.savefig('../hist' + i + '.pdf', dpi=300)