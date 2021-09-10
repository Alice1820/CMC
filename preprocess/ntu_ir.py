import os
import sys
import cv2

def my_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

root = '/mnt/hdd/NTU_RGBD_120/nturgbd_rgb'
output = '/mnt/hdd/NTU_RGBD_120_processed/nturgbd_rgb/rgb_310*256'
# os.mkdir(output)
for file1 in sorted(os.listdir(root)):
    path1 = os.path.join(root, file1)
    if os.path.isdir(path1):
        path1 = os.path.join(path1, os.listdir(path1)[0], 'nturgb+d_rgb')
        print (path1)
        for file2 in sorted(os.listdir(path1)): # .avi
            input_path = os.path.join(path1, file2)
            output_path = os.path.join(output, file2)
            os.system("ffmpeg -threads 16 -i %s -vcodec ffv1 -vf scale=310:256 %s" % (input_path, output_path))
            print (output_path)