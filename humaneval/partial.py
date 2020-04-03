__author__ = 'thiagocastroferreira'

import json

from random import shuffle

if __name__ == '__main__':
    entries = json.load(open('trials/coling/gold.json', encoding='utf-8'))
    entries = list(entries)

    sizes = set([e['size'] for e in entries])
    categories = set([e['category'] for e in entries])

    ids = []
    for category in categories:
        for size in sizes:
            f = [e for e in entries if e['size'] == size and e['category'] == category]
            shuffle(f)
            if len(f) > 0 and f[0]['eid'] not in ids:
                ids.append(f[0]['eid'])
                # print(f[0]['eid'], category, size)

    ids = sorted(ids, key=lambda x: float(x.replace('Id', '')))
    for id in ids:
        print(id)