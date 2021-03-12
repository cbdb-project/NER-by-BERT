import json
with open('data/result.json') as jf:
    gold_data = json.load(jf)

import numpy as np
person_ids = np.array(list(gold_data.keys()))


max(len(gold_data[person_id]['char_tag']) for person_id in person_ids)


import pandas as pd
jin_bio = pd.read_excel('data/Jin Bio_20190426hs.xlsx', sheet_name='bio')

test_data = []
for ix, content_without_name, _, _ in jin_bio.values:
    if isinstance(content_without_name, str) and len(content_without_name) > 0:
        test_data.append([ix, content_without_name])

test_data = pd.DataFrame(test_data, columns=['id', 'content_without_name'])

test_data.to_csv('data/test_data.txt', sep='\t', index=False)


import pandas as pd
data = pd.read_csv('data/test_data.txt', sep='\t')
data = [(str(idx), str(sent)) for idx, sent in data.values]

max(len(sent) for idx, sent in data)



# from seqeval import metrics


#############
# 抽一些 Office 的例子
import pandas as pd
office = pd.read_csv('data/office.txt', header=None)
office = office.values.reshape(1, -1)[0]
import numpy as np
a = np.random.choice(office, 11)
print('\n'.join(a))


