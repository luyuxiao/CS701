import numpy
import imageio

high_res = numpy.load('/data/yuxiao/cam_assemble/result/cam/00001985.npy', allow_pickle=True)
low_res = numpy.load('/data/yuxiao/cam1/result/cam/00006569.npy', allow_pickle=True).item()['cam'][0]
high_res = numpy.array(high_res)
low_res = numpy.array(low_res)
# temp = imageio.imread('/data/yuxiao/cam1/result/sem_seg/00006356.png')
# temp = numpy.array(temp)
imageio.imsave(('3.png'), high_res.astype(numpy.uint8))
b = 0