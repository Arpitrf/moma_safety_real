import rospy
from std_srvs.srv import Empty
rospy.wait_for_service('/clear_octomap') #this will stop your code until the clear octomap service starts running
clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)
rospy.sleep(1)

