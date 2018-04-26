r"""Predicted Gripping Point Visualization
    using a pre-trained model to detect gripping points in an image.

Example usage:
    python create_gripper_tf_record.py --logtostderr \
      --data_dir "${DATA_DIR}" \
      --tfrecord "${tfrecord}" \
      --model_dir "${model_dir}" \
      --frozen_inference_graph "${frozen_inference_graph}" \
      --label_map_path "${label_map_path}" \
      --class_num "${class_num}" \
      --output_dir_name "${output_dir_name}" \
      --is_predict "${is_predict}"


    python3 object_detection/dataset_tools/cornell_grasping_dataset_utils/visualize_predicted_gripping_point.py --logtostderr \
      --data_dir "/home/jenny/dl/gripper/data/gripper_data/" \
      --tfrecord "val.record" \
      --model_dir "/home/jenny/dl/gripper/tensorflow/detection_models/faster_rcnn_resnet101_coco_2018_01_28" \
      --frozen_inference_graph "frozen_inference_graph.pb" \
      --label_map_path "/home/jenny/dl/gripper/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt" \
      --class_num "90" \
      --output_dir_name "output_jenny_now" \
      --is_predict "n"
"""


import os
import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/jenny/dl/gripper/data/gripper_data/', 'Root directory to raw dataset.')
flags.DEFINE_string('tfrecord', 'val.record', 'TFrecord file name and extension')

flags.DEFINE_string('model_dir', '/home/jenny/dl/gripper/tensorflow/detection_models/faster_rcnn_resnet101_coco_2018_01_28', 'Root directory to pre trained model.')
flags.DEFINE_string('frozen_inference_graph', 'frozen_inference_graph.pb', 'Frozen inference graph.')
flags.DEFINE_string('label_map_path', '/home/jenny/dl/gripper/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt', 'Path to ???_label_map.pbtxt.')
flags.DEFINE_string('class_num', '90', 'Number of classes in model.')

flags.DEFINE_string('output_dir_name', 'output123456', 'Output directory name.')

flags.DEFINE_string('is_predict', 'y', 'visualize only ground truth [n] <-> run inference and visualize [y]')

FLAGS = flags.FLAGS
FLAGS.is_predict

tf.logging.set_verbosity(tf.logging.INFO)

PATH_TO_DATA_DIR = os.path.expanduser(FLAGS.data_dir)
PATH_TO_MODEL = os.path.expanduser(FLAGS.model_dir)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
# Any model exported using the `export_inference_graph.py` tool can be loaded here
# simply by changing `PATH_TO_CKPT` to point to a new .pb file.
PATH_TO_CKPT = os.path.join(PATH_TO_MODEL, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.expanduser(FLAGS.label_map_path)
NUM_CLASSES = int(FLAGS.class_num)

# Size, in inches, of the output images.
OUTPUT_IMAGE_SIZE = (12, 8)
OUTPUT_IMAGE_PATH = os.path.join(PATH_TO_DATA_DIR, FLAGS.output_dir_name)
if not os.path.isdir(OUTPUT_IMAGE_PATH):
    tf.gfile.MakeDirs(OUTPUT_IMAGE_PATH)


# # Model preparation
def _get_detection_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def _get_category_index():
    # # Loading label map (indices to category names)
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    return label_map_util.create_category_index(categories)

# # Read Data from TFRecord
def _parse_function(example_proto):
    features = {
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)}
    example = tf.parse_single_example(example_proto, features)

    image_filename = example['image/filename']
    image = tf.image.decode_png(example['image/encoded'])

    xmin = tf.expand_dims(example['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(example['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(example['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(example['image/object/bbox/ymax'].values, 0)
    bboxes = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

    bboxes = tf.expand_dims(tf.transpose(bboxes, [1, 0]), 0)

    return image_filename, image, bboxes


def _get_tensor_dict(graph):
    with graph.as_default():
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    return tensor_dict, image_tensor

category_index = None
tensor_dict = None
image_tensor = None
tf_graph = tf.get_default_graph()
if (FLAGS.is_predict=='y'):
    category_index = _get_category_index()
    tf_graph = _get_detection_graph()
    tensor_dict, image_tensor = _get_tensor_dict(tf_graph)

with tf_graph.as_default():
    tf.logging.info('Creates a dataset that reads all of the examples from files ------------>')
    dataset = tf.data.TFRecordDataset([os.path.join(PATH_TO_DATA_DIR, FLAGS.tfrecord)])
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    iterator = dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        threads_coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=threads_coord, sess=sess)
        tf.logging.info('Starting to iterate the data ------------>')
        while not threads_coord.should_stop():
            try:
                image_filename_v, image_v, bboxes_v = sess.run(iterator.get_next())
                tf.logging.info('%s was extracted.', image_filename_v.decode("utf-8"))
                tf.logging.info('bboxes: /n%s', bboxes_v)

                if (FLAGS.is_predict == 'y'): # Run inference
                    output_dict_v = sess.run([tensor_dict], feed_dict={image_tensor: np.expand_dims(image_v, 0)})[0]

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict_v['num_detections'] = int(output_dict_v['num_detections'][0])
                    output_dict_v['detection_classes'] = output_dict_v['detection_classes'][0].astype(np.uint8)
                    output_dict_v['detection_boxes'] = output_dict_v['detection_boxes'][0]
                    output_dict_v['detection_scores'] = output_dict_v['detection_scores'][0]

                    # Visualization of the results of a detection:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image=image_v,
                        boxes=output_dict_v['detection_boxes'],
                        classes=output_dict_v['detection_classes'],
                        scores=output_dict_v['detection_scores'],
                        category_index=category_index,
                        use_normalized_coordinates=True,
                        line_thickness=1,
                        skip_scores=False,
                        skip_labels=False)
                    plt.figure(figsize=OUTPUT_IMAGE_SIZE)

                # groundtruth box visualization:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image=image_v,
                    boxes=bboxes_v[0],
                    classes=[],
                    scores=None,
                    category_index={},
                    use_normalized_coordinates=True,
                    line_thickness=1)
                plt.figure(figsize=OUTPUT_IMAGE_SIZE)

                img_output_path = os.path.join(OUTPUT_IMAGE_PATH, image_filename_v.decode("utf-8"))
                tf.logging.info('Writing image WITH BOXES to disk: %s', img_output_path)
                plt.imsave(img_output_path, image_v)

                tf.logging.info('----------------------')
                plt.close()

            except tf.errors.OutOfRangeError as error:
                threads_coord.request_stop(error)

        threads_coord.request_stop()
        threads_coord.join(threads)