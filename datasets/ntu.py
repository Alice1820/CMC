import torch
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset
import random
import cv2
from sklearn.utils import shuffle

from torchvision import transforms
import torchvision.transforms as transforms
import random
import cv2
from sklearn.utils import shuffle

from PIL import ImageFilter
import random
from scipy.stats import norm

# %% tools
def load_video(path, vid_len=32):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Init the numpy array
    video = np.zeros((vid_len, height, width, 3)).astype(np.float32)
    taken = np.linspace(0, num_frames, vid_len).astype(int)

    np_idx = 0
    for fr_idx in range(num_frames):
        ret, frame = cap.read()

        if cap.isOpened() and fr_idx in taken:
            if frame is not None:
                video[np_idx, :, :, :] = frame.astype(np.float32)
            np_idx += 1

    cap.release()

    return video

def load_depth(path, vid_len=32):
    img_list = os.listdir(path)
    num_frames = len(img_list)
    width = 310
    height = 256
    dim = (width, height)
    # Init the numpy array
    video = np.zeros((vid_len, height, width)).astype(np.float32)
    taken = np.linspace(0, num_frames, vid_len).astype(int)

    np_idx = 0
    for fr_idx in range(num_frames):
        if fr_idx in taken: # 24 frames
            img_path = os.path.join(path, img_list[fr_idx])
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH) # 16bit
            if not img is None: # skip empty frame
                img = cv2.resize(img, dim) # 310*256
                video[np_idx, :, :] = np.array(img, dtype=np.float32)
            np_idx += 1
    return video

# 3d coordinates cf. https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/read_skeleton_file.m for more details
def get_3D_skeleton(path):
    # Read the full content of a file
    with open(path, mode='r') as file:
        content = file.readlines()
    content = [c.strip() for c in content]

    # Nb of frames
    num_frames = int(content[0])

    # Init the numpy array
    np_xyz_coordinates = np.zeros((3, num_frames, 25, 2)).astype(np.float32)
    # Loop over the frames
    i = 1
    for t in range(num_frames):
        # Number of person detected
        nb_person = int(content[i])

        # Loop over the number of person
        for p in range(nb_person):
            i = i + 2
            for j in range(25):
                # Catch the line of j
                i = i + 1
                content_j = content[i]

                # Split the line
                list_content_j = content_j.split(' ')
                list_content_j = [float(c) for c in list_content_j]
                xyz_coordinates = list_content_j[:3]
                # Add in the numpy array
                try:
                    for k in range(3):
                        np_xyz_coordinates[k, t, j, p] = xyz_coordinates[k]
                except Exception as e:
                    pass
                    # print(e)  # 3 persons e.g

        i += 1
    # Replace NaN by 0
    np_xyz_coordinates = np.nan_to_num(np_xyz_coordinates)
    return np_xyz_coordinates


# %%
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb, dep = sample['rgb'], sample['dep']
        subject_id, label = sample['subject_id'], sample['label']

        rgb = torch.from_numpy(rgb.astype(np.float32))
        T, W, H, C = rgb.size()
        rgb = rgb.view(T, 1, W, H, C)
        rgb = rgb.transpose(1, -1)
        rgb = rgb.squeeze().contiguous()

        dep = torch.from_numpy(dep.astype(np.float32))
        T, W, H, C = dep.size()
        dep = dep.view(T, 1, W, H, C)
        dep = dep.transpose(1, -1)
        dep = dep.squeeze().contiguous()

        return {'rgb': rgb,
                'dep': dep,
                'subject_id': torch.from_numpy(np.asarray(subject_id)),
                'label': torch.from_numpy(np.asarray(label))}


# %%
class NormalizeLen(object):
    """ Return a normalized number of frames. """

    def __init__(self, vid_len=(8, 32)):
        self.vid_len = vid_len

    def __call__(self, sample):
        rgb, dep = sample['rgb'], sample['dep']
        subject_id, label = sample['subject_id'], sample['label']
        if rgb.shape[0] != 1:
            num_frames_rgb = len(rgb)
            indices_rgb = np.linspace(0, num_frames_rgb - 1, self.vid_len[0]).astype(int)
            rgb = rgb[indices_rgb, :, :, :]
        
        if dep.shape[0] != 1:
            num_frames_dep = len(dep)
            indices_dep = np.linspace(0, num_frames_dep - 1, self.vid_len[0]).astype(int)
            dep = dep[indices_dep, :, :, :]

        return {'rgb': rgb,
                'dep': dep,
                'subject_id': subject_id,
                'label': label}


def interpole(data, cropped_length, vid_len):
    C, T, V, M = data.shape
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, :, :, None]
    data = F.interpolate(data, size=(vid_len, 1), mode='bilinear', align_corners=False).squeeze(dim=3).squeeze(dim=0)
    data = data.contiguous().view(C, V, M, vid_len).permute(0, 3, 1, 2).contiguous().numpy()
    return data


# %%

class CenterCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.9):
        self.p_interval = p_interval

    def __call__(self, sample):
        rgb, dep = sample['rgb'], sample['dep']
        subject_id, label = sample['subject_id'], sample['label']
        if rgb.shape[0] != 1:
            num_frames_rgb = len(rgb)
            bias = int((1 - self.p_interval) * num_frames_rgb / 2)
            rgb = rgb[bias:num_frames_rgb - bias, :, :, :]

        if dep.shape[0] != 1:
            num_frames_dep = len(dep)
            bias = int((1 - self.p_interval) * num_frames_dep / 2)
            dep = dep[bias:num_frames_dep - bias, :, :, :]

        return {'rgb': rgb,
                'dep': dep,
                'subject_id': subject_id,
                'label': label}


class AugCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.5):
        self.p_interval = p_interval

    def __call__(self, sample):
        rgb, dep = sample['rgb'], sample['dep']
        subject_id, label = sample['subject_id'], sample['label']
        ratio = (1.0 - self.p_interval * np.random.rand())
        if rgb.shape[0] != 1:
            num_frames_rgb = len(rgb)
            begin_rgb = (num_frames_rgb - int(num_frames_rgb * ratio)) // 2
            rgb = rgb[begin_rgb:(num_frames_rgb - begin_rgb), :, :, :]

        if dep.shape[0] != 1:
            num_frames_dep = len(dep)
            begin_dep = (num_frames_dep - int(num_frames_dep * ratio)) // 2
            dep = dep[begin_dep:(num_frames_dep - begin_dep), :, :, :]

        return {'rgb': rgb,
                'dep': dep,
                'subject_id': subject_id,
                'label': label}


def video_transform(self, np_clip):
    # if args.modality == "rgb" or args.modality == "both":
    # Div by 255
    np_clip /= 255.

    # Normalization
    np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # mean
    np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # std

    return np_clip

def depth_transform(self, np_clip):
    ####### depth ######
    # histogram, fit the first frame of each video to a gauss distribution
    # frame = np_clip[0, :, :]
    # data = np.reshape(frame, [frame.size])
    # data = data[(data >= 500) * (data <= 4500)] # range for skeleton detection
    # mu, std = norm.fit(data)
    # print (mu, std)
    # # select certain range
    # r_min = mu - std
    # r_max = mu + std
    # np_clip[(np_clip < r_min)] = 0.0
    # np_clip[(np_clip > r_max)] = 0.0
    # np_clip = np_clip - mu
    # np_clip = np_clip / std # -3~3
    p_min = 500.
    p_max = 4500.
    np_clip[(np_clip < p_min)] = 0.0
    np_clip[(np_clip > p_max)] = 0.0
    np_clip -= 2500.
    np_clip /= 2000.
    # repeat to BGR to fit pretrained resnet parameters
    np_clip = np.repeat(np_clip[:, :, :, np.newaxis], 3, axis=3) # 24, 310, 256, 3

    return np_clip

def get_dataloaders(args=None, stage='train'):
    import torchvision.transforms as transforms
    from datasets import ntu as d
    from torch.utils.data import DataLoader

    if stage == 'train':
        # Handle data
        transformer = transforms.Compose([AugCrop(), NormalizeLen((args.num_segments, args.num_segments)), ToTensor()])
    else:
        transformer = transforms.Compose([NormalizeLen((args.num_segments, args.num_segments)), ToTensor()])
    
    dataset = NTUV2(args.data_folder, transform=transformer, stage=stage, args=args)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 drop_last=False, pin_memory=True)
    n_data = len(dataset)
    print('number of samples: {}'.format(n_data))

    return dataloader, n_data

# %%
class NTU(Dataset):

    def __init__(self, root_dir='',  # /data0/xifan/NTU_RGBD_60
                 split='cross_subject', # 40 subject, 3 camera
                 stage='train',
                 vid_len=(8, 8),
                 vid_dim=256,
                 vid_fr=30,
                 args=None):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        basename_rgb = os.path.join(root_dir, 'nturgbd_rgb/avi_310x256_30') 
        basename_dep = os.path.join(root_dir, 'nturgbd_depth_masked/dep_310*256')
    
        self.vid_len = vid_len

        self.rgb_list = []
        self.dep_list = []
        self.ske_list = []
        self.labels = []
        self.subject_ids = []
        self.camera_ids = []

        if split == 'cross_subject':
            if stage == 'train':
                subjects = [1, 4, 8, 13, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
            elif stage == 'train5':
                subjects = [1]
            elif stage == 'train25':
                subjects = [1, 4, 8, 13]
            elif stage == 'train50':
                subjects = [1, 4, 8, 13, 15, 16, 17, 18]
            elif stage == 'trainexp':
                subjects = [1, 4, 8, 13, 15, 17, 19]
            elif stage == 'test':
                subjects = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
            elif stage == 'dev':  # smaller train datase for exploration
                subjects = [2, 5, 9, 14]
            else:
                raise Exception('wrong stage: ' + stage)
            self.rgb_list += [os.path.join(basename_rgb, f) for f in sorted(os.listdir(basename_rgb)) if
                          f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]
            self.dep_list += [os.path.join(basename_dep, f) for f in sorted(os.listdir(basename_dep)) if int(f[9:12]) in subjects]
            self.labels += [int(f[17:20]) for f in sorted(os.listdir(basename_rgb)) if
                        f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]
            self.subject_ids += [int(f[9:12]) for f in sorted(os.listdir(basename_rgb)) if
                        f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]
            self.camera_ids += [int(f[5:8]) for f in sorted(os.listdir(basename_rgb)) if
                        f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]
        elif split == 'cross_view':
            if stage == 'train':
                cameras = [2, 3]
            elif stage == 'trainss':  # self-supervised training
                cameras = [2, 3]
                # cameras = [3]
            elif stage == 'trains':
                cameras = [2]
            elif stage == 'test':
                cameras = [1]
            else:
                raise Exception('wrong stage: ' + stage)
            self.rgb_list += [os.path.join(basename_rgb, f) for f in sorted(os.listdir(basename_rgb)) if 
                            f.split(".")[-1] == "avi" and int(f[5:8]) in cameras]
            self.dep_list += [os.path.join(basename_dep, f) for f in sorted(os.listdir(basename_dep)) if int(f[5:8]) in cameras]
            self.labels += [int(f[17:20]) for f in sorted(os.listdir(basename)) if int(f[5:8]) in cameras]
        else:
            raise Exception('wrong mode: ' + args.mode)
        # basename_dep = os.path.join(root_dir, 'nturgbd_depth_masked/310x256_{1}')
        # basename_ske = os.path.join(root_dir, 'nturgbd_skeletons')

        # self.original_w, self.original_h = 1920, 1080

        # if args.no_bad_skel:
        #     with open("bad_skel.txt", "r") as f:
        #         for line in f.readlines():
        #             if os.path.join(basename_ske, line[:-1] + ".skeleton") in self.ske_list:
        #                 i = self.ske_list.index(os.path.join(basename_ske, line[:-1] + ".skeleton"))
        #                 self.ske_list.pop(i)
        #                 self.rgb_list.pop(i)
        #                 self.labels.pop(i)

        self.rgb_list, self.dep_list, self.labels, self.subject_ids, self.camera_ids = \
                            shuffle(self.rgb_list, self.dep_list, self.labels, self.subject_ids, self.camera_ids)

        self.transform = (stage in ['train', 'train25', 'train50', 'trainss', 'trains'])
        # self.transform = (stage in [])
        self.root_dir = root_dir
        self.stage = stage
        self.mode = stage

        self.args = args

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]
        deppath = self.dep_list[idx]

        label = self.labels[idx]
        subject_id = self.subject_ids[idx]
        camera_id = self.camera_ids[idx]

        video = np.zeros([1])
        maps = np.zeros([1])
        # skeleton = np.zeros([1])

        # if self.args.modality == "rgb" or self.args.modality == "both":
        video = load_video(rgbpath)
        depth = load_depth(deppath)
        # if self.args.modality == "skeleton" or self.args.modality == "both":
        #     skeleton = get_3D_skeleton(skepath)

        # video, maps = self.video_transform(self.args, video, maps)
        video = self.video_transform(video) # (32, 256, 310, 3)
        depth = self.depth_transform(depth)
        # print (video.shape)
        # print (depth.shape)
        if self.transform:
            video = self.augmentation(video)
            depth = self.augmentation(depth)
        else:
            video = self.transformation(video)
            # print (video.size()) # [8, 3, 224, 224]
            depth = self.transformation(depth)
        sample = {'rgb': video, 'dep': depth, 'label': label - 1, 'subject_id': subject_id - 1}
        
        # print (torch.max(depth), 'torch max')
        # print (torch.min(depth), 'torch min')
        # print (torch.mean(depth), 'torch mean')

        return sample, idx

    def augmentation(self, video):
        video = self.NormalizeLen(video, self.vid_len[0])
        video = self.NumToTensor(video)
        video = self.ToPILImage(video)
        video = self.SpaRandomFlip(video)
        video = self.SpaRandomCrop(video)
        video = self.ToTensor(video)
        return video

    def transformation(self, video):
        video = self.NormalizeLen(video, self.vid_len[0])
        video = self.NumToTensor(video)
        video = self.ToPILImage(video)
        video = self.SpaCenterCrop(video)
        video = self.ToTensor(video)
        return video

    def video_transform(self, np_clip):
        # if args.modality == "rgb" or args.modality == "both":
        # Div by 255
        np_clip /= 255.

        # Normalization
        np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # mean
        np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # std

        return np_clip

    def depth_transform(self, np_clip):
        ####### depth ######
        # histogram, fit the first frame of each video to a gauss distribution
        # frame = np_clip[0, :, :]
        # data = np.reshape(frame, [frame.size])
        # data = data[(data >= 500) * (data <= 4500)] # range for skeleton detection
        # mu, std = norm.fit(data)
        # print (mu, std)
        # # select certain range
        # r_min = mu - std
        # r_max = mu + std
        # np_clip[(np_clip < r_min)] = 0.0
        # np_clip[(np_clip > r_max)] = 0.0
        # np_clip = np_clip - mu
        # np_clip = np_clip / std # -3~3
        # print (np.max(np_clip), 'max')
        # print (np.min(np_clip), 'min')
        # print (np.mean(np_clip), 'mean')
        p_min = 500.
        p_max = 4500.
        np_clip[(np_clip < p_min)] = 0.0
        np_clip[(np_clip > p_max)] = 0.0
        np_clip -= 2500.
        np_clip /= 2000.
        # print (np.max(np_clip), 'max')
        # print (np.min(np_clip), 'min')
        # print (np.mean(np_clip), 'mean')
        # repeat to BGR to fit pretrained resnet parameters
        np_clip = np.repeat(np_clip[:, :, :, np.newaxis], 3, axis=3) # 24, 310, 256, 3

        return np_clip
    
    def TemAugCrop(self, vid):
        p_interval = 0.5
        ratio = (1.0 - p_interval * np.random.rand())
        if vid.shape[0] != 1:
            num_frames = len(vid)
            begin = (num_frames - int(num_frames * ratio)) // 2
            vid = vid[begin:(num_frames - begin), :, :, :]
        return vid

    def NormalizeLen(self, vid, vid_len=8):
        if vid.shape[0] != 1:
            num_frames = len(vid)
            indices = np.linspace(0, num_frames - 1, vid_len).astype(int)
            vid = vid[indices, :, :, :]
        return vid

    def NumToTensor(self, vid):
        return torch.from_numpy(vid.astype(np.float32)).unsqueeze(1).transpose(1, 4).squeeze()

    def SpaRandomFlip(self, vid):
        p_hflip = np.random.rand(1)
        if p_hflip > 0.5:
            return [transforms.functional.hflip(x) for x in vid]
        else:
            return vid

    def SpaRandomCrop(self, vid, vid_dim=(224, 224)):
        transform_crop = transforms.RandomCrop(vid_dim)
        coors = transform_crop.get_params(vid[0], vid_dim)
        i, j, h, w = coors
        return [transforms.functional.crop(x, i, j, h, w) for x in vid]

    def SpaCenterCrop(self, vid, vid_dim=(224, 224)):
        return [transforms.CenterCrop(vid_dim)(x) for x in vid]

    def ToPILImage(self, vid):
        return [transforms.ToPILImage()(x) for x in vid]

    def ToTensor(self, vid):
        return torch.stack([transforms.ToTensor()(x) for x in vid])


class NTUV2(Dataset):

    def __init__(self, root_dir='',  # /data0/xifan/NTU_RGBD_60
                 split='cross_subject', # 40 subject, 3 camera
                 stage='train',
                 transform=None,
                 vid_len=(8, 8),
                 vid_dim=256,
                 vid_fr=30,
                 args=None):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        basename_rgb = os.path.join(root_dir, 'nturgbd_rgb/avi_310x256_30') 
        basename_dep = os.path.join(root_dir, 'nturgbd_depth_masked/dep_310*256')
    
        self.vid_len = vid_len

        self.rgb_list = []
        self.dep_list = []
        self.ske_list = []
        self.labels = []
        self.subject_ids = []
        self.camera_ids = []

        if split == 'cross_subject':
            if stage == 'train':
                subjects = [1, 4, 8, 13, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
            elif stage == 'train5':
                subjects = [1]
            elif stage == 'train25':
                subjects = [1, 4, 8, 13]
            elif stage == 'train50':
                subjects = [1, 4, 8, 13, 15, 16, 17, 18]
            elif stage == 'trainexp':
                subjects = [1, 4, 8, 13, 15, 17, 19]
            elif stage == 'test':
                subjects = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
            elif stage == 'dev':  # smaller train datase for exploration
                subjects = [2, 5, 9, 14]
            else:
                raise Exception('wrong stage: ' + stage)
            self.rgb_list += [os.path.join(basename_rgb, f) for f in sorted(os.listdir(basename_rgb)) if
                          f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]
            self.dep_list += [os.path.join(basename_dep, f) for f in sorted(os.listdir(basename_dep)) if int(f[9:12]) in subjects]
            self.labels += [int(f[17:20]) for f in sorted(os.listdir(basename_rgb)) if
                        f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]
            self.subject_ids += [int(f[9:12]) for f in sorted(os.listdir(basename_rgb)) if
                        f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]
            self.camera_ids += [int(f[5:8]) for f in sorted(os.listdir(basename_rgb)) if
                        f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]
        elif split == 'cross_view':
            if stage == 'train':
                cameras = [2, 3]
            elif stage == 'trainss':  # self-supervised training
                cameras = [2, 3]
                # cameras = [3]
            elif stage == 'trains':
                cameras = [2]
            elif stage == 'test':
                cameras = [1]
            else:
                raise Exception('wrong stage: ' + stage)
            self.rgb_list += [os.path.join(basename_rgb, f) for f in sorted(os.listdir(basename_rgb)) if 
                            f.split(".")[-1] == "avi" and int(f[5:8]) in cameras]
            self.dep_list += [os.path.join(basename_dep, f) for f in sorted(os.listdir(basename_dep)) if int(f[5:8]) in cameras]
            self.labels += [int(f[17:20]) for f in sorted(os.listdir(basename)) if int(f[5:8]) in cameras]
        else:
            raise Exception('wrong mode: ' + args.mode)
        # basename_dep = os.path.join(root_dir, 'nturgbd_depth_masked/310x256_{1}')
        # basename_ske = os.path.join(root_dir, 'nturgbd_skeletons')

        # self.original_w, self.original_h = 1920, 1080

        # if args.no_bad_skel:
        #     with open("bad_skel.txt", "r") as f:
        #         for line in f.readlines():
        #             if os.path.join(basename_ske, line[:-1] + ".skeleton") in self.ske_list:
        #                 i = self.ske_list.index(os.path.join(basename_ske, line[:-1] + ".skeleton"))
        #                 self.ske_list.pop(i)
        #                 self.rgb_list.pop(i)
        #                 self.labels.pop(i)

        self.rgb_list, self.dep_list, self.labels, self.subject_ids, self.camera_ids = \
                            shuffle(self.rgb_list, self.dep_list, self.labels, self.subject_ids, self.camera_ids)

        self.transform = transform
        self.root_dir = root_dir
        self.stage = stage
        self.args = args

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]
        deppath = self.dep_list[idx]

        label = self.labels[idx]
        subject_id = self.subject_ids[idx]
        camera_id = self.camera_ids[idx]

        video = np.zeros([1])
        maps = np.zeros([1])
        # skeleton = np.zeros([1])

        # if self.args.modality == "rgb" or self.args.modality == "both":
        video = load_video(rgbpath)
        depth = load_depth(deppath)
        # if self.args.modality == "skeleton" or self.args.modality == "both":
        #     skeleton = get_3D_skeleton(skepath)

        # video, maps = self.video_transform(self.args, video, maps)
        video = self.video_transform(video) # (32, 256, 310, 3)
        depth = self.depth_transform(depth)
        # print (video.shape)
        # print (depth.shape)
        sample = {'rgb': video, 'dep': depth, 'label': label - 1, 'subject_id': subject_id - 1}
        
        # print (torch.max(depth), 'torch max')
        # print (torch.min(depth), 'torch min')
        # print (torch.mean(depth), 'torch mean')
        if self.transform:
            sample = self.transform(sample)

        # print (torch.max(depth), 'torch max')
        # print (torch.min(depth), 'torch min')
        # print (torch.mean(depth), 'torch mean')
        return sample, idx

    def video_transform(self, np_clip):
        # if args.modality == "rgb" or args.modality == "both":
        # Div by 255
        np_clip /= 255.

        # Normalization
        np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # mean
        np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # std

        return np_clip

    def depth_transform(self, np_clip):
        ####### depth ######
        # histogram, fit the first frame of each video to a gauss distribution
        # frame = np_clip[0, :, :]
        # data = np.reshape(frame, [frame.size])
        # data = data[(data >= 500) * (data <= 4500)] # range for skeleton detection
        # mu, std = norm.fit(data)
        # print (mu, std)
        # # select certain range
        # r_min = mu - std
        # r_max = mu + std
        # np_clip[(np_clip < r_min)] = 0.0
        # np_clip[(np_clip > r_max)] = 0.0
        # np_clip = np_clip - mu
        # np_clip = np_clip / std # -3~3
        # print (np.max(np_clip), 'max')
        # print (np.min(np_clip), 'min')
        # print (np.mean(np_clip), 'mean')
        p_min = 500.
        p_max = 4500.
        np_clip[(np_clip < p_min)] = 0.0
        np_clip[(np_clip > p_max)] = 0.0
        np_clip -= 2500.
        np_clip /= 2000.
        # print (np.max(np_clip), 'max')
        # print (np.min(np_clip), 'min')
        # print (np.mean(np_clip), 'mean')
        # repeat to BGR to fit pretrained resnet parameters
        np_clip = np.repeat(np_clip[:, :, :, np.newaxis], 3, axis=3) # 24, 310, 256, 3

        return np_clip

# %%
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", action="store",
                        dest="folder",
                        help="Path to the data",
                        default="NTU")
    parser.add_argument('--outputdir', type=str, help='output base dir', default='checkpoints/')
    parser.add_argument('--datadir', type=str, help='data directory', default='NTU')
    parser.add_argument("--j", action="store", default=12, dest="num_workers", type=int,
                        help="Num of workers for dataset preprocessing ")

    parser.add_argument("--vid_dim", action="store", default=256, dest="vid_dim",
                        help="frame side dimension (square image assumed) ")
    parser.add_argument("--vid_fr", action="store", default=30, dest="vi_fr", help="video frame rate")
    parser.add_argument("--vid_len", action="store", default=(8, 8), dest="vid_len", type=int, help="length of video")
    parser.add_argument('--modality', type=str, help='modality: rgb, skeleton, both', default='rgb')
    parser.add_argument("--hp", action="store_true", default=False, dest="hp", help="random search on hp")
    parser.add_argument("--no_norm", action="store_true", default=False, dest="no_norm",
                        help="Not normalizing the skeleton")

    parser.add_argument('--num_classes', type=int, help='output dimension', default=60)
    parser.add_argument('--batchsize', type=int, help='batch size', default=8)
    parser.add_argument("--clip", action="store", default=None, dest="clip", type=float,
                        help="if using gradient clipping")
    parser.add_argument("--lr", action="store", default=0.001, dest="learning_rate", type=float,
                        help="initial learning rate")
    parser.add_argument("--lr_decay", action="store_true", default=False, dest="lr_decay",
                        help="learning rate exponential decay")
    parser.add_argument("--drpt", action="store", default=0.5, dest="drpt", type=float, help="dropout")
    parser.add_argument('--epochs', type=int, help='training epochs', default=10)

    args = parser.parse_args()
    import torchvision.transforms as transforms

    transformer = transforms.Compose([NormalizeLen(), ToTensor()])
    train_transformer = transforms.Compose([NormalizeLen(), ToTensor()])
    dataset = NTU(args.folder, train_transformer, 'train', 32, args=args)
    iterator = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batchsize, shuffle=True,
                                           num_workers=args.num_workers)

    for batch in iterator:
        # print(batch["label"])
        print("ske", batch['ske'].shape, ", rgb", batch['rgb'].shape, ", label", batch['label'].shape)

        # print(batch["ske"])
        # check_skel(batch["ske"])
