import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_index(task):
    file1 = task + '_complex_con_verbose.npz'
    file2 = task + '_complex_bce_verbose.npz'
    con = np.load(file1)
    bce = np.load(file2)

    con_pos = con['arr_0']
    con_neg = con['arr_1']
    bce_pos = bce['arr_0']
    bce_neg = bce['arr_1']
    x = [0.1* i for i in range(0,len(con_pos))]
    plt.figure(figsize=(10,8),dpi=300)
    plt.plot(x, con_pos, color = 'r', label='BCE+ConR', marker='o', markersize=20)
    plt.plot(x, bce_pos, color = 'k', label='BCE', marker='*', markersize=20)
    plt.legend(fontsize=25)
    plt.xlabel('score', fontsize=30)
    plt.ylabel('ratio', fontsize=30)
    plt.tick_params(labelsize=20)
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.savefig(task+'_pos_score_distribution')
    plt.savefig(task+'_pos_score_distribution.pdf')
    plt.close()
    
    plt.figure(figsize=(10,8),dpi=300)
    plt.plot(x, con_neg, color = 'r', label='BCE+ConR', marker='o', markersize=20)
    plt.plot(x, bce_neg, color = 'k', label='BCE', marker='*', markersize=20)
    plt.legend(fontsize=25)
    plt.xlabel('score', fontsize=30)
    plt.ylabel('ratio', fontsize=30)
    plt.tick_params(labelsize=20)
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.savefig(task+'_neg_score_distribution')
    plt.savefig(task+'_neg_score_distribution.pdf')
    plt.close()
    print('done')

plot_index('wn')
plot_index('fb')