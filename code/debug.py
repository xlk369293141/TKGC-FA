import pickle
import os
import numpy as np

entities_to_id = {}
relations_to_id = {}
timestamps_to_id = {}
file_path = os.path.join('/home/LAB/xiaolk/TKGC-Temp/data/ICEWS14', 'ent_id')
to_read = open(file_path, 'r')
for line in to_read.readlines():
    v = line.strip().split('\t')
    nei, id = v
    entities_to_id[nei] = id
to_read.close()
file_path = os.path.join('/home/LAB/xiaolk/TKGC-Temp/data/ICEWS14', 'rel_id')
to_read = open(file_path, 'r')
for line in to_read.readlines():
    v = line.strip().split('\t')
    nei, id = v
    relations_to_id[nei] = id
to_read.close()
file_path = os.path.join('/home/LAB/xiaolk/TKGC-Temp/data/ICEWS14', 'time_id')
to_read = open(file_path, 'r')
for line in to_read.readlines():
    v = line.strip().split('\t')
    nei, id = v
    timestamps_to_id[nei] = id
to_read.close()
        
for f in ['test1', 'test2']:
    file_path = os.path.join('/home/LAB/xiaolk/TKGC-Temp/data/ICEWS14', f)
    to_read = open(file_path, 'r')
    examples = []
    for line in to_read.readlines():
        v = line.strip().split('\t ')
        lhs, rel, rhs, timestamp = v
        print(timestamp)
        try:
            examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], timestamps_to_id[timestamp]])
        except ValueError:
            continue
                
    out = open('/home/LAB/xiaolk/TKGC-Temp/data/ICEWS14' + f + '.pickle', 'wb')
    pickle.dump(np.array(examples).astype('uint64'), out)
    out.close()