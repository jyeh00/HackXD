#!/usr/bin/env python
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import rospkg
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
  
# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()
# get the file path for rospy_tutorials

path_to_learn_pkg = rospack.get_path('learn_newobjects_tf_pkg')
research_module_path = os.path.join(path_to_learn_pkg,"scripts/models/research")
object_detection_module_path = os.path.join(path_to_learn_pkg,"scripts/models/research/object_detection")
sys.path.append(object_detection_module_path)

#print(sys.path)

from object_detection.utils import ops as utils_ops
from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'learned_model'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

scripts_module_path = os.path.join(path_to_learn_pkg,"scripts/")
final_path_to_ckpt = os.path.join(scripts_module_path,PATH_TO_CKPT)

# List of the strings that is used to add correct label for each box. In our case mira_robot
PATH_TO_LABELS = os.path.join(scripts_module_path, 'training/object-detection.pbtxt')

NUM_CLASSES = 1


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(final_path_to_ckpt, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
      
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for a single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframing is required to translate the mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict
  

class RosTensorFlow():
    def __init__(self):
        # Processing the variable to process only half of the frame's lower load
        self._process_this_frame = True
        self._cv_bridge = CvBridge()

        self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('result', String, queue_size=1)
        self.score_threshold = rospy.get_param('~score_threshold', 0.1)
        self.use_top_k = rospy.get_param('~use_top_k', 5)
        
        

    def callback(self, image_msg):
        if (self._process_this_frame):
            
            image_np = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
    
            # Expand dimensions since the model expects images to have shapes: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
            cv2.imshow("Image window", image_np)
            cv2.waitKey(1)
        else:
            pass
        # We invert it
        self._process_this_frame = not self._process_this_frame
        
        
        
    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('search_garbage_auto_node')
    tensor = RosTensorFlow()
    tensor.main()
