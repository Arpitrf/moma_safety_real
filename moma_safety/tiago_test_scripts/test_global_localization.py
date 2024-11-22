import os
import rospy
from moma_safety.tiago.ros_restrict import set_init_pose, set_floor_map, change_map

rospy.init_node('test_global_localization')

if __name__ == '__main__':
    floor = 2
    bld = 'ahg'
    set_floor_map(floor_num=floor, bld=bld)
    pid = change_map(floor_num=floor, bld=bld, empty=True, fork=False)
    set_init_pose(floor=floor, bld=bld)
