import rospy
import time

from safety_run import shutdown_python_programs, start_gravity_compensation, cancel_move_base, end_gravity_compensation
from moma_safety.tiago.tiago_gym import TiagoGym

def shutdown_robot(gravity_start, move_base_cancel):
    shutdown_python_programs()
    start_gravity_compensation(gravity_start)
    cancel_move_base(move_base_cancel)
    return

if __name__ == '__main__':
    rospy.init_node('safety_run_ft')

    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type=None,
        base_enabled=True,
        torso_enabled=False,
    )

    counter = 0
    while not rospy.is_shutdown():
        wrench = env.tiago.arms["right"].ft_right_sub.get_most_recent_msg()
        f = wrench["force"]
        t = wrench["torque"]
        force_sum = abs(f.x) + abs(f.y) + abs(f.z)
        torque_sum = abs(t.x) + abs(t.y) + abs(t.z)
        if counter % 50000 == 0:
            print("force_sum: ", force_sum)
            counter = 0
        # time.sleep(0.)
        if force_sum > 100:
            print("FORCE GETING TOO HIGH. STOPPING!")
            time.sleep(5)
            # open up the fingers a bit
            env.tiago.gripper["right"].step(0.4)
            # start gc
            move_base_cancel = env.tiago.base.move_base_cancel
            gravity_start = env.tiago.arms["right"].gravity_start
            gravity_end = env.tiago.arms["right"].gravity_end
            shutdown_robot(gravity_start, move_base_cancel)
        counter += 1
