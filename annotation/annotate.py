import os

if __name__ == '__main__':
    with open('../data/dev.txt') as f:
        doc = f.read().decode('utf-8')

    doc = doc.split((50*'*')+'\n')

    print('Doc size: ', len(doc))

    entries = {}

    for _entry in doc:
        entry = _entry.split('\n\n')

        try:
            _, entryId, size, semcategory = entry[0].replace('\n', '').split()

            if size not in entries:
                entries[size] = {}
            if semcategory not in entries[size]:
                entries[size][semcategory] = []
            entries[size][semcategory].append(_entry)
        except:
            print('ERROR')

    for size in sorted(entries.keys()):
        print str(size) + 'triples'
        fname = os.path.join('dev', str(size)+'triples')
        if not os.path.exists(fname):
            os.makedirs(fname)
        for semcategory in entries[size]:
            print semcategory + ': ', str(len(entries[size][semcategory]))
            fname = os.path.join('dev', str(size) + 'triples', semcategory)
            with open(fname, 'w') as f:
                f.write('**************************************************\n'.join(entries[size][semcategory]).encode('utf-8'))
        print '\n'