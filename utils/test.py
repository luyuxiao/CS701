import numpy
import csv
lines = open('../loss.txt', 'r').readlines()
step, precision, recall, f1, loss = [], [], [], [], []
for i in range(len(lines)):
    if 'step:' in lines[i]:
        temp = lines[i].strip().split(' ')
        for _ in temp:
            if '/' in _:
                step.append(_.split('/')[0])
            if 'loss' in _:
                loss.append(_.split(':')[1])
        precision.append(lines[i+1].strip().split('\t')[-1])
        recall.append(lines[i+2].strip().split('\t')[-1])
        f1.append(lines[i+3].strip().split('\t')[-1])
    i = i + 3

target = csv.writer(open('../result.csv', 'w+'))
for i in range(len(step)):
    target.writerow([step[i], loss[i], precision[i], recall[i], f1[i]])
    print(i)