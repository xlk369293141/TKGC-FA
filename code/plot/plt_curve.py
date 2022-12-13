from tkinter import font
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


font1 = {'family' : 'Italic',
'weight' : 'normal',
'size'   : 15,
}


#Curve

ax = plt.subplot()
x = [1e-5, 0.001, 0.01, 0.1, 1]
y1 = [31.02,	31.90,	32.71,	33.17,	33.32]
y2 = [32.13,	33.94,	35.91,	33.12,	32.55]
y3 = [31.34,	32.19,	33.14,	33.04,	33.65]
y4 = [61.71,	61.80,	61.65,	61.20,	60.82]
y5 = [62.02,	62.30,	62.52,	62.27,	61.93]
y6 = [61.87,	61.77,	61.98,	61.66,	61.37]



l1=plt.plot(x,y1,'r',label=r'$\Phi_0 $', marker = 'o')
l2=plt.plot(x,y2,'g',label=r'$\Phi_1 $', marker = '^')
l3=plt.plot(x,y3,'b',label=r'$\Phi_2 $', marker = 's')
ax.plot(x,y1,'r',x,y2,'g',x,y3,'b')
# plt.ticklabel_format(style='sci', scilimits=(-1,1), axis='x',useMathText=True)
ax.set_xscale('log')
plt.xlabel('lambda', fontsize=15)
plt.ylabel('MRR', fontsize=15)
plt.legend(fontsize=15)
plt.savefig('D:\计算机\代码\画图\curve_YAGO15K.pdf',dpi=300)
# ax.xlabel('', fontsize=30)
plt.show()





ax = plt.subplot()



l1=plt.plot(x,y4,'r',label=r'$\Phi_0 $', marker = 'o')
l2=plt.plot(x,y5,'g',label=r'$\Phi_1 $', marker = '^')
l3=plt.plot(x,y6,'b',label=r'$\Phi_2 $', marker = 's')
ax.plot(x,y4,'r',x,y5,'g',x,y6,'b')
# print(type(ax))
# plt.ticklabel_format(style='sci', scilimits=(-1,1), axis='x',useMathText=True)
ax.set_xscale('log')
plt.xlabel('lambda', fontsize=15)
plt.ylabel('MRR', fontsize=15)
plt.legend(fontsize=15, prop = font1)
plt.savefig('D:\计算机\代码\画图\curve_ICEWS14.pdf',dpi=300)
# ax.xlabel('', fontsize=30)
plt.show()


# x = [32, 54, 100, 200, 400]
# y1 = []