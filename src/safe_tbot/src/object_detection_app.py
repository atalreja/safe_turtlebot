#!/usr/bin/env python
import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import rospy
from safe_tbot.srv import ImageSrv, ImageSrvResponse
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from constants import homography
Q = np.linalg.inv(homography)

CWD_PATH = '/home/cc/ee106a/fa18/class/ee106a-adn/safe_turtlebot/src/safe_tbot/src/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Create a CvBridge to convert ROS messages to OpenCV images
bridge = CvBridge()

debug_int = 0
imwidth = 1920
imheight = 1080

pub = None

def apply_to_point(H, x, y):
    aug_pt = np.array([x, y, 1]).reshape((3,1))
    return np.dot(H, aug_pt)[:2]

def get_human_pos(boxes, scores, classes):
    #assumes that there is one target human in the scene, in the case of multiple humans return the position of the one with the highest score
    # in case of no humans return none
    human_inds = np.nonzero(classes==1)
    # print(human_inds)
    # print('---------')
    # print(scores[human_inds])
    # print('---------')
    # print(np.amax(scores[human_inds]))
    # print('---------')
    # print(scores == np.amax(scores[human_inds]))
    # print('---------')
    # print(boxes[np.nonzero(scores == np.amax(scores[human_inds]))])
    # print('---------')
    if len(human_inds[0]) == 0:
        return None
    else:
        ymin, xmin, ymax, xmax = boxes[np.nonzero(scores == np.amax(scores[human_inds]))][0]
        return (xmin + xmax) / 2, ymax


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # global debug_int
    # if debug_int % 100 == 0:
    #     print(boxes)
    #     print('---------')
    #     print(scores)
    #     print('---------')
    #     print(classes)
    #     debug_int += 1

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    result = get_human_pos(boxes, scores, classes)
    if result is not None:
        x,y = result
        x = imwidth * x
        y = imheight * y
        pt = (x,y)
        
        print(x,y)


        floorx = (Q[0, 0] * pt[0] + Q[0, 1] * pt[1] + Q[0, 2]) / (Q[2, 0] * pt[0] + Q[2, 1] * pt[1] + Q[2, 2]) 
        floory = (Q[1, 0] * pt[0] + Q[1, 1] * pt[1] + Q[1, 2]) / (Q[2, 0] * pt[0] + Q[2, 1] * pt[1] + Q[2, 2]) 
        published_point = Point(floorx, floory, 0)


        print(floorx, floory)
        print('--')
    else:
        print('no humaz')
    return image_np, published_point


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()


# Converts a ROS Image message to a NumPy array to be displayed by OpenCV
def ros_to_np_img(ros_img_msg):
    return np.array(bridge.imgmsg_to_cv2(ros_img_msg,'bgr8'))

if __name__ == '__main__':
    video_source = 0
    width = 480
    height = 360 
    num_workers = 4
    queue_size = 5


    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=queue_size)
    output_q = Queue(maxsize=queue_size)
    pool = Pool(num_workers, worker, (input_q, output_q))

    # video_capture = WebcamVideoStream(src=video_source,
    #                                   width=width,
    #                                   height=height).start()

    # Waits for the image service to become available
    rospy.wait_for_service('last_image')


    # Initializes the image processing node
    rospy.init_node('object_detection_node')
    pub = rospy.Publisher('human_pos', Point, queue_size=10)
  
    # Creates a function used to call the 
    # image capture service: ImageSrv is the service type
    last_image_service = rospy.ServiceProxy('last_image', ImageSrv)



    fps = FPS().start()

    while not rospy.is_shutdown():  # fps._numFrames < 120
        try:

            ros_img_msg = last_image_service().image_data

            # Convert the ROS message to a NumPy image
            frame = ros_to_np_img(ros_img_msg)


            input_q.put(frame)

            t = time.time()

            output_rgb, published_point = output_q.get()
            pub.publish(published_point)
            output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', output_rgb)
            fps.update()

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except rospy.ServiceException, e:
            print "image_process: Service call failed: %s"%e

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
