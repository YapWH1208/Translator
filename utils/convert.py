input_file_path = 'Dataset.txt'
output_file_path = 'Dataset.xml'

with open(input_file_path, 'r', encoding='utf-8') as input_file:
    lines = input_file.readlines()

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in lines:
        parts = line.strip().split('\t')
        output_file.write(f'<en>{parts[0]}</en>\n<zh>{parts[1]}</zh>\n\n')
