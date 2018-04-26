import os
import io
import re
import random
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BboxType(Enum):
    alignedWithAxis = 1


class GripperItem:
    def __init__(self,
                 image_num, img_dir_path,
                 img_category_dict,
                 max_allowed_rotation=20,
                 img_filename_pattern='pcd{}r.png',
                 bbox_filename_pattern='pcd{}cpos.txt',
                 pcd_filename_pattern='pcd{}.txt'):

        self.image_num = image_num
        self.img_dir_path = img_dir_path
        self.img_path = os.path.join(self.img_dir_path, img_filename_pattern.format(self.image_num))

        self.format = 'png'.encode('utf8')
        self.img_filename_pattern = img_filename_pattern
        self.pcd_filename_pattern = pcd_filename_pattern

        with Image.open(self.img_path) as im:
            self.image_width, self.image_height = im.size

        self.bboxes = []
        self.categories_text = []  # List of string class/categories name of bounding box (1 per box)
        self.categories = []       # List of integer class/categories id of bounding box (1 per box)

        boxes_path = os.path.join(img_dir_path, bbox_filename_pattern.format(self.image_num))
        with open(boxes_path, "r") as boxes_file:
            lines = [line.strip().split(" ") for line in boxes_file.readlines()]
        boxes = [[float(y) for y in x] for x in lines]
        for i in range(0, len(boxes), 4):
            box_i = np.array(boxes[i:i + 4])
            if not np.isnan(box_i).any():
                theta = int(GripperItem.get_bbox_rotation(box_i))
                if not (max_allowed_rotation < theta < (90 - max_allowed_rotation) or
                        (90 + max_allowed_rotation) < theta < (180 - max_allowed_rotation)):
                    self.bboxes.append(box_i)

                    category_num, category_text = img_category_dict.get(self.image_num, (1, 'gripper'))
                    self.categories_text.append(category_text.encode('utf8'))
                    self.categories.append(int(category_num))
                else:
                    logger.info('Excluding bbox of image %s as the rotation is too big (theta = %d).', self.image_num, theta)
            else:
                logger.info('Excluding bbox of image %s as it contains Nan points.', self.image_num)

    @staticmethod
    def get_bbox_rotation(bbox):
        """
        The following text is copied from data documentation
        (http://pr.cs.cornell.edu/grasping/rect_data/readmeRawData.txt):
          Grasping rectangle files contain 4 lines for each rectangle.
          Each line contains the x and y coordinate of a vertex of that rectangle separated by a space.
          The first two coordinates of a rectangle define the line representing the orientation of the gripper plate.
          Vertices are listed in counter-clockwise order.

        :param coords:
        :return:
        """
        robot_p1 = ((bbox[1][0], bbox[1][1]), (bbox[2][0], bbox[2][1]))
        robot_p2 = ((bbox[0][0], bbox[0][1]), (bbox[3][0], bbox[3][1]))
        robot_p1_center = ((robot_p1[0][0] + robot_p1[1][0]) / 2, (robot_p1[0][1] + robot_p1[1][1]) / 2)
        robot_p2_center = ((robot_p2[0][0] + robot_p2[1][0]) / 2, (robot_p2[0][1] + robot_p2[1][1]) / 2)

        center = ((robot_p1_center[0] + robot_p2_center[0]) / 2, (robot_p1_center[1] + robot_p2_center[1]) / 2)

        robot_above_x = robot_p2_center
        if robot_p1_center[1] > center[1]:
            robot_above_x = robot_p1_center

        a = math.fabs(robot_above_x[1] - center[1])
        b = math.fabs(robot_above_x[0] - center[0])

        if robot_above_x[0] > center[0]:
            theta = math.degrees(math.atan2(a, b))
        else:
            theta = math.degrees(math.atan2(b, a)) + 90

        return theta

    def _get_bboxes_axis_aligned(self):
        def aligne_bbox (bbox):
            min_p = np.min(bbox, axis=0)
            max_p = np.max(bbox, axis=0)
            aligne = np.array([[min_p[0], max_p[1]],
                              [max_p[0], max_p[1]],
                              [max_p[0], min_p[1]],
                              [min_p[0], min_p[1]]])
            return aligne

        return list(map(aligne_bbox, self.bboxes))

    def _get_bboxes_normalized(self, bboxes):
        '''
        Normalizes each bbox with respect to image height & width.
        :param bboxes:
        :return: Dict of lists of normalized coordinates in bounding box (1 per box)
        '''
        xmins = list(map(lambda bbox: bbox[0][0]  / self.image_width, bboxes))
        ymaxs = list(map(lambda bbox: bbox[0][1] / self.image_height, bboxes))

        xmaxs = list(map(lambda bbox: bbox[3][0] / self.image_width, bboxes))
        ymins = list(map(lambda bbox: bbox[3][1] / self.image_height, bboxes))

        return {'xmins': xmins, 'xmaxs': xmaxs, 'ymins': ymins, 'ymaxs': ymaxs}

    def get_bboxes(self, bboxType):
        bboxes = {}
        if BboxType.alignedWithAxis:
            aligne = self._get_bboxes_axis_aligned()
            bboxes = self._get_bboxes_normalized(aligne)

        return bboxes

    def _get_img_depth_from_pcd(self):
        """
        The following text is copied from data documentation
        (http://pr.cs.cornell.edu/grasping/rect_data/readmeRawData.txt):
          Point cloud files Point cloud files are in .PCD v.7 point cloud data file format
          See http://www.pointclouds.org/documentation/tutorials/pcd_file_format.php for more information.
          Each uncommented line represents a pixel in the image.
          That point in space that intersects that pixel -
          has x, y, and z coordinates (relative to the base of the robot that was taking the images,
          so for our purposes we call this "global space").

          You can tell which pixel each line refers to by the final column in each line (labelled "index").
          That number is an encoding of the row and column number of the pixel.
          In all of our images, there are 640 columns and 480 rows.
          Use the following formulas to map an index to a row, col pair.
          Note that index = 0 maps to row 1, col 1.

          row = floor(index / 640) + 1
          col = (index MOD 640) + 1

        :param pcd_path:
        :param img_widht:
        :param img_height:
        :return:
        """
        pcd_path = os.path.join(self.img_dir_path, self.pcd_filename_pattern.format(self.image_num))
        with open(pcd_path, "r") as pcd_file:
            lines = [line.strip().split(" ") for line in pcd_file.readlines()]

        is_data = False
        img_depth = np.zeros((self.image_height, self.image_width), dtype='f8')
        for line in lines:
            if line[0] == 'DATA':  # skip the header
                is_data = True
                continue
            if is_data:
                d = float(line[2])
                i = int(line[4])
                col = i % self.image_height
                row = math.floor(i / self.image_width)
                img_depth[row, col] = d

        min_d = np.min(img_depth)
        max_d = np.max(img_depth)
        max_min_diff = max_d - min_d

        def normalize(x):
            return 255 * (x - min_d) / max_min_diff

        normalize = np.vectorize(normalize, otypes=[np.float])
        img_depth = normalize(img_depth)
        return img_depth

    def get_image_encoded(self, depth_img_output_path=None):
        img_depth = self._get_img_depth_from_pcd()
        img_depth = Image.fromarray(img_depth).convert('L')
        with Image.open(self.img_path) as im:
            red_channel, green_channel, _ = im.split()
            #  Replacing the blue channel in the image with the depth information:
            im_with_depth = Image.merge("RGB", (red_channel, green_channel, img_depth))

        with io.BytesIO() as imgByteArr:
            im_with_depth.save(imgByteArr, format='PNG')
            img = imgByteArr.getvalue()

        if depth_img_output_path:
            img_depth.save(os.path.join(depth_img_output_path, '{}_depth.png'.format(self.image_num)))
            im_with_depth.save(os.path.join(depth_img_output_path, '{}_depthInBlue.png'.format(self.image_num)))

        return img

    def draw_grip_bboxes(self, img_output_path):
        bboxes_dict = {'org':          (self.bboxes, 'Crimson'),
                       'axis aligned': (self._get_bboxes_axis_aligned(), 'Gold')}
        with Image.open(os.path.join(self.img_dir_path, self.img_filename_pattern.format(self.image_num))) as image:
            draw = ImageDraw.Draw(image)
            i = 1
            for box_type in bboxes_dict:
                bboxes = bboxes_dict[box_type][0]  #TODO
                color = bboxes_dict[box_type][1]  #TODO
                for bbox in bboxes:
                    draw.line((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]), fill=color, width=1)
                    draw.line((bbox[1][0], bbox[1][1], bbox[2][0], bbox[2][1]), fill=color, width=1)
                    draw.line((bbox[2][0], bbox[2][1], bbox[3][0], bbox[3][1]), fill=color, width=1)
                    draw.line((bbox[3][0], bbox[3][1], bbox[0][0], bbox[0][1]), fill=color, width=1)

                draw.text((self.image_width / 3, 30 + 20 * i), box_type, color, font=ImageFont.load_default())
                i += 1
            if not os.path.isdir(img_output_path):
                os.mkdir(img_output_path)

            image.save(os.path.join(img_output_path, '{}_bboxes.png'.format(self.image_num)))


class GripperDataset:
    def __init__(self, data_path, output_path,
                 img_filename_pattern='pcd{}r.png',
                 bbox_filename_pattern='pcd{}cpos.txt',
                 pcd_filename_pattern='pcd{}.txt',
                 img_category_filepath='processedData/z.txt',
                 train_dataset_percentage=0.8, val_dataset_percentage=0.2, test_dataset_percentage=0,
                 max_allowed_rotation=20,
                 aggregate_labels='gripper',
                 draw_grip_rectangles=False):
        '''
        Initializes the self.dataset dics to contain:
         - ['images_dataset']: image path -> train/test/val set
         - ['images']: image number -> item
         - ['annotations']: image number -> list of bbox
         - ['categories']: image number -> list of categories
        :param FLAGS:
        :param annotation_file:
        '''
        self.dataset_types = {}  # key:type / value: (percentage_min, percentage_max)
        self.images, images = {}, {}                    # image path -> item
        self.annotations, annotations = {}, {}         # image number -> list of bbox
        self.categories, categories = {}, {}           # image number -> list of categories
        self.categories_text, categories_text = {}, {}
        self.images_dataset, images_dataset = {}, {}    # image -> train/test/val set

        ## input check:
        assert data_path, '`data_dir` is missing.'
        assert os.path.isdir(data_path), '`data_path` is not a directory.'
        data_path = os.path.expanduser(data_path)
        logger.info('Reading data from: %s', data_path)

        assert len(img_filename_pattern) > 0 and "{}" in img_filename_pattern, '`img_filename_pattern` is a valid pattern - should contain `{}` to match image numbers.'
        assert len(bbox_filename_pattern) > 0 and "{}" in bbox_filename_pattern, '`bbox_filename_pattern` is a valid pattern - should contain `{}` to match image numbers.'
        assert len(pcd_filename_pattern) > 0 and "{}" in pcd_filename_pattern, '`pcd_filename_pattern` is a valid pattern - should contain `{}` to match image numbers.'

        img_category_path = os.path.join(data_path, img_category_filepath)
        assert os.path.isfile(img_category_path), '`img_category_filepath`' + img_category_path + ' is not a file.'

        def is_between_0_and_1(num):
            try:
                num = float(num)
                return 0 <= num <= 1
            except ValueError:
                return False

        assert is_between_0_and_1(train_dataset_percentage), '`train_dataset_percentage` is not a float [0,1].'
        assert is_between_0_and_1(val_dataset_percentage), '`val_dataset_percentage` is not a float [0,1].'
        assert is_between_0_and_1(test_dataset_percentage), '`test_dataset_percentage` is not a float [0,1].'
        train_dataset_percentage = float(train_dataset_percentage)
        val_dataset_percentage = float(val_dataset_percentage)
        test_dataset_percentage = float(test_dataset_percentage)
        assert train_dataset_percentage + \
               val_dataset_percentage + \
               test_dataset_percentage, '`{train, val, test}_dataset_percentage` should sum to 1.'

        def set_dataset(type, dataset_percentage, acc):
            if dataset_percentage > 0.0:
                self.dataset_types[type] = (acc, acc + dataset_percentage)
                acc += dataset_percentage
            return acc

        acc = set_dataset('train', train_dataset_percentage, 0)
        acc = set_dataset('val', val_dataset_percentage, acc)
        _ = set_dataset('test', test_dataset_percentage, acc)

        assert 'gripper' == aggregate_labels or \
               os.path.isfile(aggregate_labels) or \
               'no' == aggregate_labels, '`aggregate_labels` should be one of {`gripper`, path to aggregation file, `no`}.'

        assert output_path, '`output_path` is missing.'
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        output_path = os.path.expanduser(output_path)


        def assign_image_to_dataset(img_num):
            if item:  # Might be empty/None in case there are no 'good' bbox in this image
                rand = random.uniform(0, 1)
                for dataset_type, (dataset_p_min, dataset_p_max) in self.dataset_types.items():
                    if dataset_p_min < rand <= dataset_p_max:
                        logger.info('[random number %.2f]: Writing img %s -> %s set.', rand, img_num, dataset_type)
                        images_dataset[img_num] = dataset_type
                        break

        img_category_dict = GripperDataset.create_img_category_dict(data_path, output_path, aggregate_labels)

        for rootpath, dirnames, filenames in os.walk(data_path):
            regex = re.compile(img_filename_pattern.format('[0-9]*')) #TODO chack re.findall: regex = re.compile(r'_x\d+_y\d+\.npy') img_list = filter(regex.search, filenames)
            img_list = filter(regex.search, filenames)
            for img in img_list:
                img_num = re.findall(r'\d+', img)[0]
                item = GripperItem(img_num, rootpath,
                                   img_category_dict,
                                   max_allowed_rotation,
                                   img_filename_pattern, bbox_filename_pattern, pcd_filename_pattern)
                bboxes = item.get_bboxes(BboxType.alignedWithAxis)
                if bboxes:
                    images[img_num] = item
                    annotations[img_num] = bboxes
                    categories[img_num] = item.categories
                    categories_text[img_num] = item.categories_text
                    assign_image_to_dataset(img)
                else:
                    logger.info('Excluding image %s as it contains no valid bboxes.', img_num)

                if draw_grip_rectangles: item.draw_grip_bboxes(output_path)

        if images_dataset:
            for data_set_type in self.dataset_types.keys():
                logger.info('%d images in %s dataset.', len(list(filter(lambda t: t==data_set_type, images_dataset.values()))), data_set_type)
        else:
            logger.info('No Images in dataset.')

        self.images = images
        self.annotations = annotations
        self.categories = categories
        self.categories_text = categories_text
        self.images_dataset = images_dataset


    @staticmethod
    def create_img_category_dict(data_dir_path,
                                 output_path,
                                 aggregate_labels='gripper'):
        def load_class_dict(aggregation_file_path):
            '''
            :param aggregation_file_path:
            :return: dictionary with ->
              key: org_class_name /
              value: (class_num, aggregated_class_name)
            '''
            logger.info('Loading class aggregation file from %s', aggregation_file_path)
            with open(aggregation_file_path, "r") as aggregation_file:
                lines = [re.split(', |,| |\n', line.strip()) for line in
                         aggregation_file.readlines()]  # TODO split very sensitive
            agg_class_dict = {}
            for next_line_num in range(0, len(lines), 5):
                agg_class_name = lines[next_line_num + 2][1]
                class_num = lines[next_line_num + 1][1]
                for org_class_name in lines[next_line_num + 3][1:]:
                    agg_class_dict[org_class_name] = (int(class_num), agg_class_name)

            return agg_class_dict

        def load_img_category_file():
            with open(os.path.join(data_dir_path, 'processedData/z.txt'), "r") as class_files:
                lines = [line.strip().split(" ") for line in class_files.readlines()]
            return lines

        img_class_map = {}  # key: org_class_name / value: (class_num, class_name)
        class_dict = {}
        if aggregate_labels == 'gripper':
            class_dict['*'] = (1, 'gripper')
            img_class_map['*'] = (1, 'gripper')
        elif aggregate_labels == 'no':
            lines = load_img_category_file()
            class_num = 1

            for line in lines:
                img_num = line[0]
                if img_num not in img_class_map:
                    class_name = line[2]
                    if not class_name in class_dict:
                        class_dict[class_name] = (class_num, class_name)
                        class_num += 1
                    img_class_map[img_num] = (class_dict[class_name][0], class_name)
        else:  # with aggregation to classes
            lines = load_img_category_file()
            class_dict = load_class_dict(os.path.expanduser(aggregate_labels))

            for line in lines:
                img_num = line[0]
                if img_num not in img_class_map:
                    org_class_name = line[2]
                    img_class_map[img_num] = class_dict[org_class_name]

        gripper_label_map_file = os.path.join(output_path, 'gripper_label_map.txt')
        logger.info('Creating label map filename at %s', gripper_label_map_file)
        item_str = "item {{\n  id: {}\n  name: '{}'\n}}\n"
        with open(gripper_label_map_file, "w") as label_map_file:
            for item in sorted(set(class_dict.values())):
                label_map_file.write(item_str.format(item[0], item[1]))

        return img_class_map

    def get_dataset(self, dataset_type='traning'):
        if type in self.dataset_types.keys():
            image_ids = list(filter(lambda t: t == dataset_type, self.images_dataset.values()))

            dataset = {
                'images': {img_num: self.images[img_num] for img_num in image_ids},
                'annotations': {img_num: self.annotations[img_num] for img_num in image_ids},
                'categories': {img_num: self.categories[img_num] for img_num in image_ids},
                'categories_text': {img_num: self.categories_text[img_num] for img_num in image_ids},
            }
        else:
            dataset = {}
        return dataset



# gripper_dataset1 = GripperDataset('/home/jenny/dl/gripper/data/gripper_data/raw_dataset',
#                                  '/home/jenny/test03',
#                                  aggregate_labels='/home/jenny/gripper/cornell_grasping_dataset_tools/gripper_label_map.txt')
#
#
#
# for item in gripper_dataset1.dataset['images'].values():
#     item.get_image_encoded('/home/jenny/gripper/test01')