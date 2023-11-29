import os
import re
import json
import random
import config
DEVICE = config.device


def build_dataset(xml_folder, train_data_path, dev_data_path, test_data_path, max_length, prob=0.85):
    data = []
    
    # Read data from XML files
    for file in os.listdir(xml_folder):
        if file.endswith('.xml'):
            with open(f'{xml_folder}/{file}', 'r', encoding='utf-8') as f:
                content = f.read()
                en_lines = re.findall(r'<en>(.*?)</en>', content)
                zh_lines = re.findall(r'<zh>(.*?)</zh>', content)
                
                i = 0
                while i < len(en_lines):
                    if zh_lines[i] == '':
                        i += 1
                        continue
                    en_line = en_lines[i].replace('\\n', '\n')
                    zh_line = zh_lines[i].replace('\\n', '\n')
                    
                    if len(en_line) > max_length:
                        i += 1
                        continue
                        
                    while random.random() < prob and i+1 < len(en_lines) and zh_lines[i+1] != '' and len(en_line)+len(en_lines[i+1])+1 <= max_length:
                        i += 1
                        en_line += '\n' + en_lines[i].replace('\\n', '\n')
                        zh_line += '\n' + zh_lines[i].replace('\\n', '\n')
                    
                    data.append([en_line, zh_line])
                    i += 1

    # Shuffle the data
    random.shuffle(data)

    # Split data into train, dev, and test sets
    test_size = int(0.1 * len(data))
    dev_size = int(0.1 * len(data))
    train_data = data[:-(dev_size + test_size)]
    dev_data = data[-(dev_size + test_size):-test_size]
    test_data = data[-test_size:]

    # Write data to files
    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)

    with open(dev_data_path, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False)

    with open(test_data_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)

if __name__ == '__main__':
    build_dataset(config.xml_folder, config.train_data_path, config.dev_data_path, config.test_data_path, config.max_len)