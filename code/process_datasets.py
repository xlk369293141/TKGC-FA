import os
import errno
from pathlib import Path
import pickle
import sys
import numpy as np

from collections import defaultdict
from datetime import datetime
DATA_PATH = "../data"

def prepare_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    KGC = ['WN18RR', 'FB237', 'YAGO3-10']
    TKGC = ['ICEWS14', 'ICEWS05-15', 'GDELT', 'YAGO15K']
    if name in TKGC:
        Tag = True
    else:
        Tag = False
    files = ['train', 'valid', 'test']
    entities, relations, timestamps = set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            v = line.strip().split('\t')
            if Tag == True:
                if len(v) == 4:
                    lhs, rel, rhs, timestamp = v
                    timestamps.add(datetime.strptime(timestamp, '%Y-%m-%d'))
                elif len(v) == 3:
                    lhs, rel, rhs = v
                    rel += '_notime'
                elif len(v) == 5:
                    lhs, rel, rhs, type, timestamp = v
                    rel += type
                    timestamp = timestamp[1:5] + '-1-1'
                    timestamps.add(datetime.strptime(timestamp, '%Y-%m-%d'))
            else:
                lhs, rel, rhs = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
        to_read.close()

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    print("{} entities and {} relations".format(len(entities), len(relations)))
    if Tag == True:    
        timestamps_to_id = {x.strftime('%Y-%m-%d'): i for (i, x) in enumerate(sorted(timestamps))}
        print("{} timestamps, from {} to {}".format(len(timestamps), min(timestamps), max(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)
    
    try:
        os.makedirs(os.path.join(DATA_PATH, name))
    except OSError as e:
        r = input(f"{e}\nContinue ? [y/n]")
        if r != "y":
            sys.exit()

    # write ent to id / rel to id
    if Tag == True:
        for (dic, f) in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'time_id']):
            ff = open(os.path.join(DATA_PATH, name, f), 'w+')
            for (x, i) in dic.items():
                ff.write("{}\t{}\n".format(x, i))
            ff.close()
    else:
        for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
            ff = open(os.path.join(DATA_PATH, name, f), 'w+')
            for (x, i) in dic.items():
                ff.write("{}\t{}\n".format(x, i))
            ff.close()
    
    # ts = np.array(sorted(timestamps.keys()), dtype='float')
    # diffs = ts[1:] - ts[:-1]
    # out = open(os.path.join(DATA_PATH, name, 'ts_diffs.pickle'), 'wb')
    # pickle.dump(diffs, out)
    # out.close()
    
    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            if Tag == False:
                lhs, rel, rhs = line.strip().split('\t')
                try:
                    examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
                except ValueError:
                    continue
            else:
                v = line.strip().split('\t')
                if len(v) == 5:
                    lhs, rel, rhs, type, timestamp = v
                    rel += type
                    timestamp = timestamp[1:5] + '-1-1'
                    timestamp = datetime.strptime(timestamp, '%Y-%m-%d')
                    timestamp = timestamp.strftime('%Y-%m-%d')
                    try:
                        examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], timestamps_to_id[timestamp]])
                    except ValueError:
                        continue
                elif len(v) == 3:
                    lhs, rel, rhs = v
                    rel += '_notime'
                    try:
                        examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], len(timestamps_to_id)])
                    except ValueError:
                        continue                    
                else:
                    lhs, rel, rhs, timestamp = v
                    try:
                        examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], timestamps_to_id[timestamp]])
                    except ValueError:
                        continue
                 
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")
    # (Undetermined, how to filter candidate)
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        if Tag == False:
            for lhs, rel, rhs in examples:
                to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
                to_skip['rhs'][(lhs, rel)].add(rhs)            
        else:
            for lhs, rel, rhs, timestamp in examples:
                to_skip['lhs'][(rhs, rel + n_relations, timestamp)].add(lhs)  # reciprocals
                to_skip['rhs'][(lhs, rel, timestamp)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }
    if Tag == False:
        for lhs, rel, rhs in examples:
            counters['lhs'][lhs] += 1
            counters['rhs'][rhs] += 1
            counters['both'][lhs] += 1
            counters['both'][rhs] += 1    
    else:
        for lhs, rel, rhs, timestamp in examples:
            counters['lhs'][lhs] += 1
            counters['rhs'][rhs] += 1
            counters['both'][lhs] += 1
            counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


if __name__ == "__main__":
    # datasets = ['WN18RR', 'FB237', 'YAGO3-10']
    # datasets = ['ICEWS14', 'ICEWS05-15', 'GDELT', 'YAGO15K']  
    datasets = ['YAGO15K']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset(
                os.path.join(
                    '../src_data', d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise