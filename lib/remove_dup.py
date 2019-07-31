## Usages Python remove_dup.py input.json output.json

import sys
import pandas as pd

def remove_duplicate(data, col1):
    
    data['col_new'] = data[col1].apply(lambda x : None if x == None else ''.join(map(str, x)))
    m = data['col_new'].duplicated(keep='first')
    m = m.values
    m = [not x for x in m]
    data_new = data[m]
    data_new = data_new.drop(['col_new'], axis = 1)
    
    return data_new


f = open(sys.argv[1], 'r')
data1 = pd.read_json(f, lines=True)
col1 = 'id'
col2 = 'jobs'


data_new = remove_duplicate(data1, col1)
data_new = remove_duplicate(data_new, col2)


with open(sys.argv[2], 'w') as f1:
    f1.write(data_new.to_json(orient='records', lines=True))


f.close()
f1.close()

