import numpy as np
from scipy.fftpack import fft

import os
from pathlib import Path
import pickle

import numpy as np
import datetime
from dateutil import relativedelta
from collections import defaultdict

# Fs =1000      # 采样频率
# f1 =390       # 信号频率1
# f2 = 120    # 信号频率2
# t=np.linspace(0,1,Fs)   # 生成 1s 的实践序列   
# noise1 = 0.1 * np.random.random(Fs)      # 0-1 之间的随机噪声
# noise2 = 0.1 * np.random.normal(1,10,Fs)
# #产生的是一个10e3的高斯噪声点数组集合（均值为：1，标准差：10）
# y=2*np.sin(2*np.pi*f1*t)+5*np.sin(2*np.pi*f2*t)+noise2

# def FFT (Fs,data):
#     L = len (data)                        # 信号长度
#     N =np.int64(np.power(2,np.ceil(np.log2(L))))    # 下一个最近二次幂
#     FFT_y1 = np.abs(fft(data,N))/L*2      # N点FFT 变化,但处于信号长度
#     Fre = np.arange(int(N/2))*Fs/N        # 频率坐标
#     FFT_y1 = FFT_y1[range(int(N/2))]      # 取一半
#     return Fre, FFT_y1

# # Fre, FFT_y1 = FFT(Fs,y)
# # plt.figure()
# # plt.plot(Fre,FFT_y1)
# # plt.grid()
# # plt.savefig("./debug.png")

DATA_PATH = "../data"

def process_fft(path, name):
    TKGC = ['ICEWS14', 'ICEWS05-15', 'GDELT']
    if name in TKGC:
        Tag = True
    else:
        Tag = False
    files = ['train', 'valid', 'test']
    
    tmp = defaultdict(set)
    datelist = []
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs, year, month, day in examples:
            tmp[(lhs, rel, rhs)].add(datetime.date(year, month, day))
            datelist.append(datetime.date(year, month, day))
    oldest = min(datelist)
    youngest = max(datelist)
    delta = (youngest-oldest).days
    print(delta)
    
if __name__ == "__main__":
    # datasets = ['WN18RR', 'FB237', 'YAGO3-10']
    datasets = ['ICEWS14', 'ICEWS05-15', 'GDELT']  
    for d in datasets:
        print("Preparing dataset {}".format(d))
        process_fft(os.path.join('../src_data', d), d)
    
    