#!/usr/bin/env python
import os
import argparse

import rospy
from pal_navigation_msgs.srv import *
import moma_safety.utils.vision_utils as VU

def old_change_map():
    rospy.init_node("test_change_map")
    rospy.wait_for_service("/pal_map_manager/change_map")
    try:
        print("Acknowledgement")
        proxy = rospy.ServiceProxy("/pal_map_manager/change_map", Acknowledgment)
        # proxy = rospy.ServiceProxy("/pal_map_manager/change_map", ChangeMap)
        # response = proxy("input: '2024-05-03_2242'")
        response = proxy("input: 'ahg_full'")
        print(response)
        rospy.loginfo(response)
    except rospy.ServiceException as e:
        rospy.logerror("Service call failed: %s" % e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor", type=int, choices=[1, 2], required=True)
    parser.add_argument("--bld", type=str, choices=['ahg', 'mbb'], required=True)
    args = parser.parse_args()
    VU.set_floor_map(args.floor, bld=args.bld)
