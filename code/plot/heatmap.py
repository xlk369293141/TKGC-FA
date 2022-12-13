from matplotlib.pyplot import annotate, figure
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
import numpy as np
from palettable.lightbartlein.sequential import Blues10_10



list = [62.50, 62.33, 62.2, 62.16, 62.24, 62.51, 62.48, 62.22, 62.35, 62.42, 62.44, 62.34, 62.39]


f, ax = plt.subplots()
data=np.array(list).reshape(1, 13)
im = plt.imshow(data, cmap = Blues10_10.mpl_colormap)

# for i in range(len(data)):
#         plt.text(i, '%.2f' % list[i], s=str,
#                  horizontalalignment='center',
#                  verticalalignment='center')
for y in range(data.shape[0]):
    for x in range(data.shape[1]):
        plt.text(x, y , '%.2f' % data[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.grid(False)
f.colorbar(im, orientation = 'horizontal')
plt.xlabel('Frquency        Low → high', weight='bold', fontsize=14)
plt.yticks([])
plt.xticks([])
plt.savefig('D:\计算机\代码\画图\heatmap_ICEWS14.pdf',dpi=300)

plt.show()
plt.close()

# list = [35.62, 35.23, 35.64, 35.63, 35.48, 35.78, 35.55, 35.49, 35.90, 35.55, 35.73, 35.44, 35.56]


# f, ax = plt.subplots()
# data=np.array(list).reshape(1, 13)
# im = plt.imshow(data, cmap = Blues10_10.mpl_colormap, )

# # for i in range(len(data)):
# #         plt.text(i, '%.2f' % list[i], s=str,
# #                  horizontalalignment='center',
# #                  verticalalignment='center')
# for y in range(data.shape[0]):
#     for x in range(data.shape[1]):
#         plt.text(x, y , '%.2f' % data[y, x],
#                  horizontalalignment='center',
#                  verticalalignment='center',
#                  )
# plt.grid(False)

# f.colorbar(im, orientation = 'horizontal')



# plt.xlabel('Frquency        Low → high', weight='bold', fontsize=14)
# plt.yticks([])
# plt.xticks([])
# plt.savefig('D:\计算机\代码\画图\heatmap_YAGO15K.pdf',dpi=300)
# plt.show()
# plt.close()