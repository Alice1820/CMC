import os
import cv2

def my_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

root = '/mnt/hdd/NTU_RGBD_120/nturgbd_depth_masked'
output = '/mnt/hdd/NTU_RGBD_120_processed/nturgbd_depth_masked/dep_310*256'
# os.mkdir(output)
for file1 in sorted(os.listdir(root)):
    path1 = os.path.join(root, file1)
    if os.path.isdir(path1):
        path1 = os.path.join(path1, os.listdir(path1)[0], 'nturgb+d_depth_masked')
        print (path1)
        for file2 in sorted(os.listdir(path1)):
            path2 = os.path.join(path1, file2)
            for file3 in sorted(os.listdir(path2)):
                path3 = os.path.join(path2, file3)
                dep = cv2.imread(path3, -1)
                dep = cv2.resize(dep, [310, 256])
                output_path = os.path.join(output, file2)
                my_mkdir(output_path)
                output_path = os.path.join(output_path, file3)
                cv2.imwrite(output_path, dep)
                print (output_path)