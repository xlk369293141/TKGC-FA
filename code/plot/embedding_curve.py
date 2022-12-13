from tkinter import font
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


font1 = {'family' : 'Italic',
'weight' : 'normal',
'size'   : 15,
}


ax = plt.subplot()
x = [32, 54, 100, 200, 400]
y1 = [30.3,	32.0,	33.3,	34.7,	35.4]
y2 = [22.3,	26.2,	28.9,	34.4,	36.1]
y3 = [47.6,	51.4,	53.7,	57.2,	59.3]
y4 = [46.7,	50.9,	53.3,	57.1,	60.0]
y5 = [49.4,	53.2,	57.3,	60.3,	62.5]

l1=plt.plot(x,y1,'r', label=r'TNTComplex', marker = "o")
l2=plt.plot(x,y2,'b', label=r'TuckER_FA', marker = "^")

ax.plot(x,y1,'r',x,y2,'b')
# plt.ticklabel_format(style='sci', scilimits=(-1,1), axis='x',useMathText=True)

plt.xlabel('Embedding', fontsize=15)
plt.ylabel('MRR', fontsize=15)
plt.legend(fontsize=15, prop = font1)
plt.savefig('.\embedding_YAGO15K.pdf',dpi=300)
# ax.xlabel('', fontsize=30)
plt.show()


ax = plt.subplot()
l1=plt.plot(x,y3,'r', label=r'TNTComplex', marker = "o")
l2=plt.plot(x,y4,'g', label=r'TeLM', marker = "s")
l3=plt.plot(x,y5,'b', label=r'TuckER_FA', marker = "^")

ax.plot(x,y3,'ro',x,y4,'gs',x,y5,'b^')
# plt.ticklabel_format(style='sci', scilimits=(-1,1), axis='x',useMathText=True)

plt.xlabel('Embedding', fontsize=15)
plt.ylabel('MRR', fontsize=15)
plt.legend(fontsize=15, prop = font1)
plt.savefig('.\embedding_ICEWS14.pdf',dpi=300)
# ax.xlabel('', fontsize=30)
# plt.show()




