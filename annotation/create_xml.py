import os
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom


def parse_annotation(in_file):
    '''
    Parse an annotation document and extract references from the texts
    :param doc:
    :return:
    '''
    with open(in_file) as f:
        doc = f.read().decode("utf-8")
        doc = doc.split((50*'*')+'\n')

    data = []
    for entry in doc:
        entry = entry.split('\n\n')
        templates = []
        texts = []
        try:
            _, entryId, size, semcategory = entry[0].replace('\n', '').split()

            entity_map = dict(map(lambda entity: entity.split(' | '), entry[2].replace('\nENTITY MAP\n', '').split('\n')))

            lexEntries = entry[3].replace('\nLEX\n', '').split('\n-')[:-1]

            for lex in lexEntries:
                if lex[0] == '\n':
                    lex = lex[1:]
                lex = lex.split('\n')

                text = lex[1].replace('TEXT: ', '').strip()
                template = lex[2].replace('TEMPLATE: ', '')
                correct = lex[3].replace('CORRECT: ', '').strip()
                comment = lex[4].replace('COMMENT: ', '').strip()

                texts.append(text)

                if comment in ['g', 'good']:
                    templates.append(template)
                elif correct != '' and comment != 'wrong':
                    if correct.strip() == 'CORRECT:':
                        correct = template
                    templates.append(correct)
                else:
                    template = ''
                    templates.append(template)

            instance = {
                'docid':entryId,
                'size':size,
                'semcategory':semcategory,
                'entity_map': entity_map,
                'texts':texts,
                'templates':templates
            }
            data.append(instance)
        except:
            print('ERROR')
            print entry

    return data

def parse_corpus(annotations, in_file, out_file):
    tree = ET.parse(in_file)
    root = tree.getroot()

    entries = root.find('entries')

    for _entry in entries:
        docid = _entry.attrib['eid']
        size = int(_entry.attrib['size'])
        category = _entry.attrib['category']

        entitymap = ET.SubElement(_entry, 'entitymap')

        data = filter(lambda x: x['docid']==docid and x['size']==str(size) and x['semcategory']==category, annotations)
        if len(data) > 0:
            data = data[0]

            for tag, entity in data['entity_map'].iteritems():
                entity_xml = ET.SubElement(entitymap, 'entity')
                entity_xml.text = tag + ' | ' + entity

            # process lexical entries
            lexEntries = _entry.findall('lex')
            for i, lexEntry in enumerate(lexEntries):
                lexEntry.text = ''

                text = data['texts'][i]
                template = data['templates'][i]

                text_xml = ET.SubElement(lexEntry, 'text')
                text_xml.text = text

                template_xml = ET.SubElement(lexEntry, 'template')
                template_xml.text = template

    # tree.write(out_file, encoding='utf-8', xml_declaration=True)

    rough_string = ET.tostring(tree.getroot(), encoding='utf-8', method='xml').replace('\n', '')
    rough_string = re.sub(">[^\S\n\t]+<", '><', rough_string)
    xml = minidom.parseString(rough_string).toprettyxml(indent="\t")

    with open(out_file, 'w') as f:
        f.write(xml.encode('utf-8'))

def run(corpus_dir, annotation_dir, out_dir):
    for fname in os.listdir(corpus_dir):
        if fname != '.DS_Store':
            in_annotation_file = os.path.join(annotation_dir, fname.replace('.xml', ''))
            annotations = parse_annotation(in_annotation_file)

            in_corpus_file = os.path.join(corpus_dir, fname)
            out_file = os.path.join(out_dir, fname)
            parse_corpus(annotations, in_corpus_file, out_file)


if __name__ == '__main__':

    # TRAIN SET
    TRAIN_CORPUS_DIR = '../data/cyber/train'
    TRAIN_ANNOTATION_DIR = 'train'
    TRAIN_ANNOTATION_OUT = 'final/train'
    for dir in os.listdir(TRAIN_CORPUS_DIR):
        if dir != '.DS_Store':
            corpus_dir = os.path.join(TRAIN_CORPUS_DIR, dir)
            annotation_dir = os.path.join(TRAIN_ANNOTATION_DIR, dir)
            out_dir = os.path.join(TRAIN_ANNOTATION_OUT, dir)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            run(corpus_dir, annotation_dir, out_dir)

    # DEV SET
    DEV_CORPUS_DIR = '../data/cyber/dev'
    DEV_ANNOTATION_DIR = 'dev'
    DEV_ANNOTATION_OUT = 'final/dev'
    for dir in os.listdir(DEV_CORPUS_DIR):
        if dir != '.DS_Store':
            corpus_dir = os.path.join(DEV_CORPUS_DIR, dir)
            annotation_dir = os.path.join(DEV_ANNOTATION_DIR, dir)
            out_dir = os.path.join(DEV_ANNOTATION_OUT, dir)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            run(corpus_dir, annotation_dir, out_dir)