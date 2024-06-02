#!/usr/bin/env python3
import rospy
from robot_vision.msg import BoundingBoxes
from geometry_msgs.msg import Twist

class ObjectDetectionStopper:
    def __init__(self):
        rospy.init_node('object_detection_stopper')
        self.subscriber = rospy.Subscriber('/yolov5/targets', BoundingBoxes, self.detection_callback)
        self.publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.stop_distance_threshold = rospy.get_param('~stop_distance_threshold', 1.0)  # 停车距离阈值，默认为1米

    def detection_callback(self, msg):
        for box in msg.bounding_boxes:
            if box.distance < self.stop_distance_threshold:
                self.stop_car()
                break

    def stop_car(self):
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.publisher.publish(stop_twist)
        rospy.loginfo("Object detected. Stopping the car.")

if __name__ == '__main__':
    try:
        stopper = ObjectDetectionStopper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass