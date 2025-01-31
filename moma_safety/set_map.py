#!/usr/bin/env python
import os
import time
import atexit
import threading
import rospy
import yaml
from nav_msgs.msg import OccupancyGrid
from PIL import Image
import argparse
import numpy as np

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener

# import the message type
from geometry_msgs.msg import PoseWithCovarianceStamped

global thread_num

def create_init_pose_with_covariance_stamped(pos, ori):
    # create a Pose message without covariance
    pose = PoseWithCovarianceStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = rospy.Time.now()
    pose.pose.pose.position.x = pos[0]
    pose.pose.pose.position.y = pos[1]
    pose.pose.pose.position.z = pos[2]
    pose.pose.pose.orientation.x = ori[0]
    pose.pose.pose.orientation.y = ori[1]
    pose.pose.pose.orientation.z = ori[2]
    pose.pose.pose.orientation.w = ori[3]

    # Set the covariance matrix
    pose.pose.covariance = [0.90, 0.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892326654787]
    return pose

def set_init_pose(floor, bld, publisher=None):
    # sets the rough initial pose of the robot after using elevator in each floor and building
    print(f"Setting init_pose for floor {floor} in bld {bld}")
    if publisher is None:
        publisher = Publisher('/initialpose', PoseWithCovarianceStamped)
        rospy.sleep(3)

    if bld == 'ahg':
        if floor == 1:
            # msg = create_init_pose_with_covariance_stamped([-4.61, -9.15, 0.0], [0.0, 0.0, -1.0, 0.0])
            msg = create_init_pose_with_covariance_stamped([-2.83, 36.33, 0.0], [0.0, 0.0, 1.0, 0.0])
        elif floor == 2:
            msg = create_init_pose_with_covariance_stamped([-4.41, -8.48, 0.0], [0.0, 0.0, -1.0, 0.0])
        else:
            raise NotImplementedError
    elif bld == 'mbb':
        if floor == 2:
            msg = create_init_pose_with_covariance_stamped([-6.26, 2.25, 0.0], [0.0, 0.0, -0.7032503722560098, 0.710942271863042])
    print(msg)
    for _ in range(10):
        publisher.write(msg)
        # wait for 1 second
        rospy.sleep(0.5)
    return True

def load_map(map_yaml_path, empty=False):
    print(map_yaml_path)
    with open(map_yaml_path, 'r') as yaml_file:
        map_metadata = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    # Load the image using PIL
    image_path = map_metadata['image']
    image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)
    print("image_path", image_path)
    resolution = map_metadata['resolution']
    origin = map_metadata['origin']

    grid_data, image_shape = convert_pgm_to_occupancy(image_path, empty=empty)
    return grid_data, resolution, origin, image_shape

def convert_pgm_to_occupancy(pgm_path, empty=False):
    # Load the PGM image
    image = Image.open(pgm_path)
    # flip the image
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image = np.array(image)

    # Map image values to occupancy values
    grid_data = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i][j]
            if empty:
                grid_data.append(0)
            else:
                if pixel > 250:  # White or near-white: free space
                    grid_data.append(0)
                elif pixel < 127:  # Black or near-black: occupied space
                    grid_data.append(100)
                else:  # Gray or intermediate: mark as unknown
                    grid_data.append(-1)

    return grid_data, image.shape

def publish_map_obs(map_file, empty=False):
    # Create a publisher on the specified topic
    map_pub = rospy.Publisher('map_obs', OccupancyGrid, queue_size=1)
    # wait for the publisher to be ready
    rospy.sleep(1)

    # Load the map from a given YAML file
    grid_data, resolution, origin, dimensions = load_map(map_file, empty=empty)

    # Create OccupancyGrid message
    grid_msg = OccupancyGrid()
    grid_msg.header.frame_id = "map"  # or another appropriate frame
    grid_msg.header.stamp = rospy.Time.now()
    grid_msg.info.resolution = resolution
    grid_msg.info.width = dimensions[1]
    grid_msg.info.height = dimensions[0]
    grid_msg.info.origin.position.x = origin[0]
    grid_msg.info.origin.position.y = origin[1]
    grid_msg.info.origin.position.z = 0
    grid_msg.info.origin.orientation.w = 1  # No rotation
    grid_msg.data = grid_data

    # Publish the map at a fixed rate
    # rate = rospy.Rate(0.1)  # 0.1 Hz
    # while not rospy.is_shutdown():
    map_pub.publish(grid_msg)
    import ipdb; ipdb.set_trace()
    # rate.sleep()

    return

def kill_node():
    rospy.signal_shutdown("Shutdown")
    rospy.spin()

def cmdlineparse(args):
    parser = ArgumentParser()
    parser.add_argument('--empty', action='store_true')
    parser.add_argument('--prev_pid', type=int, default=None)
    parser.add_argument('--floor_num', type=int, default=2)
    args=parser.parse_args(args)
    return args

def change_map(floor_num, bld, empty=False, prev_pid=None, fork=False):
    map_file = None
    if bld == 'ahg':
        if floor_num == 2:
            map_file = '/home/pal/Desktop/rutav/maps/ahg2_blk/map.yaml'  # Update this path
        elif floor_num == 1:
            map_file = '/home/pal/Desktop/rutav/maps/ahg1_blk/map.yaml'  # Update this path if any restricted areas in the map
        elif floor_num == -1:
            map_file = '/home/pal/Desktop/rutav/maps/ahg_test/map.yaml'
        else:
            raise NotImplementedError
    elif bld == 'mbb':
        if floor_num == 1:
            map_file = '/home/pal/Desktop/rutav/maps/mbb_1st/map.yaml'  # Update this path
        elif floor_num == 2:
            map_file = '/home/pal/Desktop/rutav/maps/mbb_2nd_blk/map.yaml'  # Update this path if any restricted areas in the map
        elif floor_num == 3:
            map_file = '/home/pal/Desktop/rutav/maps/mbb_3rd_blk/map.yaml'
        else:
            raise NotImplementedError
    elif bld == 'nhb':
        if floor_num == 3:
            map_file = '/home/pal/Desktop/rutav/maps/nhb_3rd_blk/map.yaml'
    else:
        raise NotImplementedError

    # this is a cleaner way. However, ros handles this part for us as well.
    if prev_pid:
        os.system(f"kill -9 {prev_pid}")

    pid = 0
    if fork:
        raise NotImplementedError # did not work as expected
        # launch a process for executing publish_map_obs
        # thread = threading.Thread(target=publish_map_obs, args=(map_file, empty))
        # thread.start()
    else:
        publish_map_obs(map_file, empty=empty)

    return pid

def set_floor_map(floor_num, bld):
    if bld == 'ahg':
        if floor_num == 1:
            map_name = "ahg_1st"
        elif floor_num == 2:
            map_name = "ahg_full"
        elif floor_num == -1:
            map_name = "empty_global_map"
        else:
            raise NotImplementedError
    elif bld == 'mbb':
        if floor_num == 1:
            map_name = "mbb_1st"
        elif floor_num == 2:
            map_name = "mbb_2nd"
        elif floor_num == 3:
            map_name = "mbb_3rd"
        else:
            raise NotImplementedError
    elif bld == 'nhb':
        if floor_num == 3:
            map_name = "nhb_3rd"
    else:
        raise NotImplementedError
    print(f"Calling: rosservice call /pal_map_manager/change_map '{map_name}'")
    os.system(f"rosservice call /pal_map_manager/change_map '{map_name}'")
    # wait for 1 second
    rospy.sleep(1)
    return True

def main(args=None):
    args = cmdlineparse(args)
    change_map(args.floor_num, empty=args.empty, prev_pid=args.prev_pid, fork=False)

if __name__ == '__main__':
    rospy.init_node('map_publisher')
    time.sleep(5)
    set_floor_map(-1, 'ahg')
    # change_map(2, 'ahg', empty=True)
    # main()