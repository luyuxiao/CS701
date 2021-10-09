import os

test_dir = '/home/yuxiao/public/img_dir/test1'
images = os.listdir(test_dir)
fake_label = open('/home/yuxiao/public/fake_test1_label.txt', 'w+')
for image in images:
    fake_label.write(image + ' 0\n')