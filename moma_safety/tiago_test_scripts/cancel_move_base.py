import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction

# Initialize the ROS node
rospy.init_node('cancel_move_base_goal')

# Create an action client for the move_base action server
client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

# Wait for the action server to start
client.wait_for_server()

# Cancel all goals
client.cancel_all_goals()

print("All move_base goals have been cancelled.")