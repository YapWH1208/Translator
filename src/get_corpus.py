import json

def get_corpus():
    JSON_FILEPATH = '../data/'
    files = ['train', 'dev', 'test']
    ch_path = JSON_FILEPATH + 'corpus.ch'
    en_path = JSON_FILEPATH + 'corpus.en'
    ch_lines = []
    en_lines = []

    for file in files:
        corpus = json.load(open(JSON_FILEPATH + 'json/' + file + '.json', 'r', encoding='utf-8'))
        for item in corpus:
            ch_lines.append(item[1] + '\n')
            en_lines.append(item[0] + '\n')

    with open(ch_path, "w", encoding='utf-8') as fch:
        fch.writelines(ch_lines)

    with open(en_path, "w", encoding='utf-8') as fen:
        fen.writelines(en_lines)

    # lines of Chinese: 252777
    print("lines of Chinese: ", len(ch_lines))
    # lines of English: 252777
    print("lines of English: ", len(en_lines))
    print("-------- Get Corpus ! --------")
