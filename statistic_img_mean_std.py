import os
import numpy
from utils.Metrics import *

output = numpy.load('/home/yuxiao/CS701/temp1.npy')
pred = numpy.load('/home/yuxiao/CS701/temp2.npy')
label = numpy.load('/home/yuxiao/CS701/temp3.npy')

# number of labels
num_of_label = numpy.sum(label, axis=0)
print(num_of_label)

f1 = f1_score(pred.T, label.T)
print(f1)
print(numpy.mean(f1))

index = 10
for i in range(103):
    index += 1
    print(str(num_of_label[i]) + "\t" + str(numpy.sum(pred, axis=0)[i]) + "\t" + str(f1[i]))

for i in range(4005):
    if label[i, 19] > 0.5:
        print(i)
for i in range(4005):
    if pred[i, 19] > 0.5:
        print(i)