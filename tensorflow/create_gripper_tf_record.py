"""Convert raw "Cornell grasping dataset" (http://pr.cs.cornell.edu/grasping/rect_data/data.php) to TFRecord for object_detection.

1. Make sure to download & unpack dataset:
1.1. Raw Dataset (total fo 10 zip files) first:
     wget http://pr.cs.cornell.edu/grasping/rect_data/temp/data01.tar.gz
     tar -xvzf data01.tar.gz

1.2. ProcessedData:
     wget http://pr.cs.cornell.edu/grasping/rect_data/processedData.zip
     unzip processedData.zip -d processedData

4. create TFrecord.
Example usage:
    python create_gripper_tf_record.py --logtostderr \
      --data_dir="${DATA_DIR_PATH}" \
      --output_dir="${OUTPUT_DIR_PATH}" \
      --aggregate_labels="${gripper, path to aggregation file, no}"

    *** data_dir & output_dir are mandatory.
    python3 object_detection/dataset_tools/create_gripper_tf_record.py --logtostderr \
      --data_dir=/home/jenny/gripper2/data/gripper_data/raw_dataset/ \
      --output_dir=/home/jenny/gripper2/data/gripper_data/
      --aggregate_labels=gripper
"""

import os
from contextlib import ExitStack
import tensorflow as tf
from object_detection.utils import dataset_util
from graspingAPI import gripper

print(tf.__version__)

flags = tf.app.flags

flags.DEFINE_string('data_dir', '', 'Root directory to raw dataset.')

flags.DEFINE_string('aggregate_labels', 'gripper', '{gripper, path to aggregation file, no}')
#flags.DEFINE_string('aggregate_labels', '/home/jenny/dl/gripper/tensorflow/models/research/object_detection/data/gripper_aggregtion_map.txt', '{gripper, path to aggregation file, no}')
#flags.DEFINE_string('aggregate_labels', 'no', '{gripper, path to aggregation file, no}')

flags.DEFINE_string('output_dir', '', 'Path to a directory to output the created TFRecords.')

FLAGS = flags.FLAGS


def main(_):
    print("dddddddddddddddddddd")
    input_path = FLAGS.input_path
    output_path = FLAGS.output_path
    dataset = gripper.GripperDataset(FLAGS.input_path, FLAGS.output_path)

    images = dataset.dataset['images']
    images_dataset = dataset.dataset['images_dataset']
    annotations = dataset.dataset['annotations']
    categories = dataset.dataset['categories']
    categories_text = dataset.dataset['categories_text']

    output_path = os.path.expanduser(FLAGS.output_dir)
    with ExitStack() as stack:
        tf_record_writers = {}

        for image_num, item in images:
            item.get_image_encoded()

            bytes_filename = item.image_num.encode('utf8')
            img = item.get_image_encoded()
            img_annotations = annotations[image_num]
            img_categories = categories[image_num]
            img_categories_text = categories_text[image_num]
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(item.image_height),
                'image/width': dataset_util.int64_feature(item.image_width),
                'image/filename': dataset_util.bytes_feature(bytes_filename),
                'image/source_id': dataset_util.bytes_feature(bytes_filename),
                'image/encoded': dataset_util.bytes_feature(img),
                'image/format': dataset_util.bytes_feature(item.format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(img_annotations.get('xmins')),
                'image/object/bbox/xmax': dataset_util.float_list_feature(img_annotations.get('xmaxs')),
                'image/object/bbox/ymin': dataset_util.float_list_feature(img_annotations.get('ymins')),
                'image/object/bbox/ymax': dataset_util.float_list_feature(img_annotations.get('ymaxs')),
                'image/object/class/text': dataset_util.bytes_list_feature(img_categories_text),
                'image/object/class/label': dataset_util.int64_list_feature(img_categories),
            }))

            if not tf_record_writers.get(images_dataset[image_num]):
                tf_record_writers[images_dataset[image_num]] = tf.python_io.TFRecordWriter(
                    os.path.join(output_path, images_dataset[image_num] + '.record'))

            tf_record_writers[images_dataset[image_num]].write(tf_example.SerializeToString())
    print("11111111111111111")


if __name__ == '__main__':
    tf.app.run()
