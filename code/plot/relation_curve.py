from tkinter import font
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

font1 = {'family' : 'Italic',
'weight' : 'normal',
'size'   : 15,
}


#Curve

ax = plt.subplot()
x = np.arange(2,16,1)
y1 = [7.2314, 11.7352, 13.3811, 6.1235, 7.9640, 8.0121, 8.2682, 6.0065, 5.0945, 8.2211, 6.7001, 5.3062, 5.6956, 4.7068]
y2 = [10.9519, 7.7477, 7.7332, 13.6062, 13.5742, 8.7979, 9.7264, 9.6326, 10.4677, 7.9776, 8.0792, 8.2602, 9.3441, 7.0865]
y3 = [8.1156, 7.3805, 9.6264, 7.1505, 8.1525, 9.3769, 10.0027, 8.3368, 7.9877, 12.6117, 12.6476, 8.2742, 7.6676, 10.4371]
y4 = [7.3337, 8.0824, 8.4643, 7.3137, 7.5538, 8.8331, 7.8865, 8.6213, 7.9026, 7.8080, 8.4114, 8.7003, 8.1831, 6.7550]
x_ticks_labels = [2011,2012,2015,2016]

X_Y_Spline_1 = make_interp_spline(x, y1)
X_Y_Spline_2 = make_interp_spline(x, y2)
X_Y_Spline_3 = make_interp_spline(x, y3)
X_Y_Spline_4 = make_interp_spline(x, y4)
X_ = np.linspace(x.min(), x.max(), 500)
Y1 = X_Y_Spline_1(X_)
Y2 = X_Y_Spline_2(X_)
Y3 = X_Y_Spline_3(X_)
Y4 = X_Y_Spline_4(X_)


l1=plt.plot(X_,Y1,'g',label=r'Oldham Athletic F.C.')
l2=plt.plot(X_,Y2,'b',label=r'Brentford F.C.')
l3=plt.plot(X_,Y3,'orange',label=r'Leeds United F.C.')
l4=plt.plot(X_,Y4,'dimgray',label=r'Rotherham United F.C.')




ax.plot(X_,Y1,'g',X_,Y2,'b',X_,Y3,'orange',X_,Y4,'dimgray')
ax.set_xticks([3.5,5.5,11.5,13.5])
ax.set_xticklabels(x_ticks_labels, rotation='45')

plt.axvspan(3, 4, facecolor='greenyellow', alpha=0.5)
plt.axvspan(5, 6, facecolor='cornflowerblue', alpha=0.5)
plt.axvspan(11, 12, facecolor='y', alpha=0.5)
plt.axvspan(13, 14, facecolor='gray', alpha=0.5)


# ax.set_facecolor('greenyellow')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Score', fontsize=15)
# plt.xticks([3.5, 5.5, 11.5, 13.5])

plt.yticks(rotation=0)
plt.legend(fontsize=15)
plt.savefig('D:\计算机\代码\画图\curve_relation.pdf',dpi=300)
# ax.xlabel('', fontsize=30)
plt.show()