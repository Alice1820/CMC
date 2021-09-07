import os
import sys
import cv2

def my_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

root = '/mnt/hdd/NTU_RGBD_60/nturgbd_ir'
output = '/mnt/hdd/NTU_RGBD_60_processed/nturgbd_ir/ir_310*256'
# os.mkdir(output)
for file1 in sorted(os.listdir(root)):
    path1 = os.path.join(root, file1)
    if os.path.isdir(path1):
        path1 = os.path.join(path1, 'nturgb+d_ir')
        print (path1)
        for file2 in sorted(os.listdir(path1)): # .avi
            input_path = os.path.join(path1, file2)
            output_path = os.path.join(output, file2)
            os.system("ffmpeg -i %s -vcodec ffv1 -vf scale=310:256 %s" % (input_path, output_path))
            print (output_path)