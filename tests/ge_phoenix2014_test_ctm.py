from spacy.tokens.span_group import deepcopy

if __name__ == '__main__':
    stm_lines = []

    with open('../data/global_files/ground_truth/phoenix2014-groundtruth-train.stm', 'r') as f:
        for line in f:
            stm_lines.append(line.strip())
    with open('../data/global_files/ground_truth/phoenix2014-groundtruth-dev.stm', 'r') as f:
        for line in f:
            stm_lines.append(line.strip())
    with open('../data/global_files/ground_truth/phoenix2014-groundtruth-test.stm', 'r') as f:
        for line in f:
            stm_lines.append(line.strip())

    stm_lines = [line.split() for line in stm_lines if line]

    with open('./sssss.stm', 'w') as f:
        for line in stm_lines:
            f.write(' '.join(line) + '\n')

    ctm_lines = []
    for line in stm_lines:
        for i in range(5, len(line)):
            ctm_line = deepcopy(line[0:5])
            ctm_line.append(line[i])
            ctm_lines.append(ctm_line)


    with open('./phoenix2014_test_ctm1.ctm', 'w') as f:
        for line in ctm_lines:
            f.write(' '.join(line) + '\n')
