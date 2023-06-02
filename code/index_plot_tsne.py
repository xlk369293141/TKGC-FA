import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold


X1 = np.load('../logs/TuckER_ATT_L4_ICEWS14_10_64504/time_embedding.npy')
X2 = np.load('../logs/TuckER_ATT_L4_YAGO15K_10_64670/time_embedding.npy')
n1 = np.size(X1, 0)
n2 = np.size(X2, 0)

print('start t-SNE')
# for i in [5, 10, 30 , 50 ,75, 100]:
perp = 30
tsne = manifold.TSNE(n_components=2, perplexity=perp, method='exact', init='pca')

t0 = time.time()
Y1 = tsne.fit_transform(X1)
t1=time.time()

y1 = np.array([i for i in range(len(X1))])
norm = plt.Normalize(y1.min(), y1.max())
norm_y1 = norm(y1)
print(len(y1))
fig = plt.figure(figsize=(8, 8))
# color = ['b','g','r','m','y','k','tan','purple']
plt.plot(Y1[:,0], Y1[:,1])
plt.scatter(Y1[:,0], Y1[:,1], c=norm_y1, cmap='viridis', s=100)
# plt.scatter(Y1[:,0], Y1[:,1], color='purple', s=100)
plt.title("t-SNE-ICEWS14 {} sec  - {} perplexity".format(t1-t0, perp))
# plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig("./plot/t-SNE-ICEWS14-{}".format(perp))
print('half done')

Y2 = tsne.fit_transform(X2)
t2=time.time()
y2 = np.array([i for i in range(len(X2))])
norm = plt.Normalize(y2.min(), y2.max())
norm_y2 = norm(y2)
# print(len(y2))
fig = plt.figure(figsize=(8, 8))
plt.plot(Y2[:, 0], Y2[:, 1])
plt.scatter(Y2[:, 0], Y2[:, 1], c=norm_y2, cmap='viridis',s=100)
# plt.scatter(Y2[:, 0], Y2[:, 1], color='purple', s=100)
plt.title("t-SNE-YAGO15K {} sec - {} perplexity".format(t2-t1, perp))
plt.savefig("./plot/t-SNE-YAGO15K-{}".format(perp))

print('done')