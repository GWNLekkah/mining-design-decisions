from dl_manager.accelerator import *
from dl_manager.feature_generators.util import text_cleaner
import json

with open('../datasets/issuedata/all_issues_raw.json') as file:
    docs = []
    for issue in json.load(file):
        docs.append(issue['summary'])
        docs.append(issue['description'])


N = 100

import time
print('Bulk')
start = time.time()
bulk_clean_text_parallel(docs[:N], 'markers', 6)
print('Took', time.time() - start)

print('Native')
start = time.time()
for d in docs[:N]:
    text_cleaner.remove_formatting(d, text_cleaner.FormattingHandling.Markers)
print('Took', time.time() - start)