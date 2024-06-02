#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import torch
import numpy as np
import message_filters
from functools import partial
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from robot_vision.msg import BoundingBox, BoundingBoxes

class Yolov5Param:
    def __init__(self):
        yolov5_path = rospy.get_param('/yolov5_path', '')
        weight_path = rospy.get_param('~weight_path', '')
        conf = rospy.get_param('~conf', '0.5')
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local', force_reload=True)
        if rospy.get_param('/use_cpu', 'false'):
            self.model.cpu()
        else:
            self.model.cuda()
        self.model.conf = conf

        self.target_pub = rospy.Publisher("/yolov5/targets", BoundingBoxes, queue_size=1)

def image_depth_cb(rgb_msg, depth_msg, cv_bridge, yolov5_param, color_classes, image_pub):
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image = cv_bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        frame = np.array(cv_image, dtype=np.uint8)
    except CvBridgeError as e:
        print(e)
        return
    
    bounding_boxes = BoundingBoxes()
    bounding_boxes.header = rgb_msg.header

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolov5_param.model(rgb_frame)
    
    boxes = results.pandas().xyxy[0].values
    for box in boxes:
        xmin, ymin, xmax, ymax, confidence, class_name = box[:6]
        bounding_box = BoundingBox()
        bounding_box.probability = float(confidence)
        bounding_box.xmin = int(xmin)
        bounding_box.ymin = int(ymin)
        bounding_box.xmax = int(xmax)
        bounding_box.ymax = int(ymax)
        bounding_box.Class = box[-1]
        
        # 提取深度信息
        depth_roi = depth_image[int(ymin):int(ymax), int(xmin):int(xmax)]
        valid_depths = depth_roi[depth_roi > 0]
        # 如果存在有效深度值，找到最小值；否则设置为 nan
        if valid_depths.size > 0:
            distance = np.min(valid_depths)
        else:
            distance = float('nan')
        
        bounding_box.distance = distance
        bounding_boxes.bounding_boxes.append(bounding_box)
        
        if class_name in color_classes.keys():
            color = color_classes[class_name]
        else:
            color = np.random.randint(0, 183, 3)
            color_classes[class_name] = color
    
        cv2.rectangle(cv_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (int(color[0]), int(color[1]), int(color[2])), 2)    
        text_pos_y = ymin + 30 if ymin < 20 else ymin - 10
        # cv2.putText(cv_image, class_name, (int(xmin), int(text_pos_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"{box[-1]} ({distance:.2f} m)", (int(box[0]), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    
    yolov5_param.target_pub.publish(bounding_boxes)
    image_pub.publish(cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))

def main():
    rospy.init_node("yolov5_detector")
    rospy.loginfo("starting yolov5_detector node")

    bridge = CvBridge()
    image_pub = rospy.Publisher("/yolov5/detection_image", Image, queue_size=1)
    
    yolov5_param = Yolov5Param()
    color_classes = {}
    
    bind_image_depth_cb = partial(image_depth_cb, cv_bridge=bridge, yolov5_param=yolov5_param, color_classes=color_classes, image_pub=image_pub)

    rgb_sub = message_filters.Subscriber("/camera/image_raw", Image)
    depth_sub = message_filters.Subscriber("/kinect/depth/image_raw", Image)

    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
    ts.registerCallback(bind_image_depth_cb)
    
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
