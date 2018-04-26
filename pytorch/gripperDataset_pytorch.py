import torch.utils.data as data
from PIL import Image
import os
import os.path
from graspingAPI import gripper


class GripperDetection(data.Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, full_dataset, set_name=None, transform=None, target_transform=None):
        self.grip_dataset = full_dataset.get_dataset(set_name)
        self.images = self.grip_dataset['images']
        self.ids = list(self.images.keys())
        self.images_dataset = self.grip_dataset['images_dataset']
        self.annotations = self.grip_dataset['annotations']
        self.categories = self.grip_dataset['categories']
        self.categories_text = self.grip_dataset['categories_text']

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        target = self.annotations[img_id]

        path = self.images[img_id].img_path
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


#
# gripper_dataset1 = GripperDetection('/home/jenny/dl/gripper/data/gripper_data/raw_dataset',
#                                     '/home/jenny/gripper/test113')
#
# print('aaa')
#
#
#
# for item in gripper_dataset1.dataset['images'].values():
#     item.get_image_encoded('/home/jenny/gripper/test01')