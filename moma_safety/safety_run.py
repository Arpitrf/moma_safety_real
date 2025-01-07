import os
import  psutil
from pynput import mouse
from pynput import keyboard

import rospy
from pal_common_msgs.msg import EmptyActionGoal
from actionlib_msgs.msg import GoalID

def start_gravity_compensation(gravity_start):
    print("Starting gravity compensation")
    rospy.sleep(0.1)
    msg = EmptyActionGoal()
    gravity_start.publish(msg)
    print("Gravity compensation started")
    return

def cancel_move_base(move_base_cancel):
    print("Cancelling move_base")
    rospy.sleep(0.1)
    msg = GoalID()
    move_base_cancel.publish(msg)
    print("move_base cancelled")
    return

def shutdown_python_programs():
    print("Shutting down python programs")
    current_pid = os.getpid()
    for process in psutil.process_iter(['pid', 'name']):
        try:
            # Check if the process name is 'python' or 'python3' and it's not the current process
            if process.info['name'] in ['python', 'python3'] and process.info['pid'] != current_pid:
                process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return

def end_gravity_compensation(gravity_end):
    print("Ending gravity compensation")
    rospy.sleep(0.1)
    msg = GoalID()
    gravity_end.publish(msg)
    print("Gravity compensation ended")
    return

# Define the functions to be called
def on_left_double_click(gravity_start, move_base_cancel):
    print("Left button double-clicked!")
    shutdown_python_programs()
    start_gravity_compensation(gravity_start)
    cancel_move_base(move_base_cancel)
    return

def on_right_double_click(gravity_end):
    print("Right button double-clicked!")
    end_gravity_compensation(gravity_end)
    return

def on_middle_double_click(move_base_cancel):
    cancel_move_base(move_base_cancel)
    return

# Create a listener class to monitor mouse clicks
class MouseClickListener:
    def __init__(self):
        self.left_clicks = 0
        self.right_clicks = 0
        self.left_click_start_time = rospy.Time.now()
        self.right_click_start_time = rospy.Time.now()
        self.middle_button_start_time = rospy.Time.now()
        self.middle_button_clicks = 0
        self.reset_time = 0.5
        self.n_clicks = 4
        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()

        self.gravity_start = rospy.Publisher('/gravity_compensation/goal', EmptyActionGoal, queue_size=1)
        self.gravity_end = rospy.Publisher('/gravity_compensation/cancel', GoalID, queue_size=1)
        self.move_base_cancel = rospy.Publisher('/move_base/cancel', GoalID, queue_size=1)
        rospy.sleep(2.0)

    def on_click(self, x, y, button, pressed):
        if pressed:
            print(f"Mouse clicked at ({x}, {y}) with button {button}")
            if rospy.Time.now() - self.left_click_start_time > rospy.Duration.from_sec(self.reset_time):
                self.left_clicks = 0
            if rospy.Time.now() - self.right_click_start_time > rospy.Duration.from_sec(self.reset_time):
                self.right_clicks = 0
            if rospy.Time.now() - self.middle_button_start_time > rospy.Duration.from_sec(self.reset_time):
                self.middle_button_clicks = 0

            if button == mouse.Button.left:
                self.left_clicks += 1
                if (self.left_clicks == 1):
                    self.left_click_start_time = rospy.Time.now()
                elif self.left_clicks == self.n_clicks:
                    on_left_double_click(gravity_start=self.gravity_start, move_base_cancel=self.move_base_cancel)
                    self.left_clicks = 0  # Reset the click counter
            elif button == mouse.Button.right:
                self.right_clicks += 1
                if (self.right_clicks == 1):
                    self.right_click_start_time = rospy.Time.now()
                elif self.right_clicks == self.n_clicks:
                    on_right_double_click(gravity_end=self.gravity_end)
                    self.right_clicks = 0
            elif button == mouse.Button.middle:
                self.middle_button_clicks += 1
                if self.middle_button_clicks == 1:
                    self.middle_button_start_time = rospy.Time.now()
                elif self.middle_button_clicks == 2:
                    print("Middle button clicked!")
                    on_middle_double_click(self.move_base_cancel)
                    self.middle_button_clicks = 0
        else:
            # Reset the counters if button is released
            if rospy.Time.now() - self.left_click_start_time > rospy.Duration.from_sec(self.reset_time):
                self.left_clicks = 0
            if rospy.Time.now() - self.right_click_start_time > rospy.Duration.from_sec(self.reset_time):
                self.right_clicks = 0
            if rospy.Time.now() - self.middle_button_start_time > rospy.Duration.from_sec(self.reset_time):
                self.middle_button_clicks = 0

    def run(self):
        self.listener.join()

    def start(self):
        self.listener.start()

# Puckjs listener should setup as a keyboard listener for the puckjs buttons
class PuckjsListener:
    def __init__(self):
        self.state = 0
        # mapping of the puckjs buttons to the keyboard keys; it acts as a keyboard emulator
        # single click puckjs is mapped to: "abc"
        # double click puckjs is mapped to: "def" without first click
        # triple click puckjs is mapped to: "xyz"
        # 1 click is start gravity compensation, cancel move_base, shutdown python programs
        # 2 clicks without first click is cancel move_base
        # 3 clicks is end gravity compensation
        # setup keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.gravity_start = rospy.Publisher('/gravity_compensation/goal', EmptyActionGoal, queue_size=1)
        self.gravity_end = rospy.Publisher('/gravity_compensation/cancel', GoalID, queue_size=1)
        self.move_base_cancel = rospy.Publisher('/move_base/cancel', GoalID, queue_size=1)
        rospy.sleep(1.0)

    def on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            return
        k = key.char
        if k == 'a':
            self.state = 1
        elif (k == 'b') and (self.state == 1):
            self.state = 2
        elif (k == 'c') and (self.state == 2):
            self.state = 3
        elif k == 'd':
            self.state = 4
        elif (k == 'e') and (self.state == 4):
            self.state = 5
        elif (k == 'f') and (self.state == 5):
            self.state = 6
        elif k == 'x':
            self.state = 7
        elif (k == 'y') and (self.state == 7):
            self.state = 8
        elif (k == 'z') and (self.state == 8):
            self.state = 9
        if self.state == 3:
            print("Emergency exits!")
            start_gravity_compensation(self.gravity_start)
            cancel_move_base(self.move_base_cancel)
            shutdown_python_programs()
            self.state = 0
        elif self.state == 6:
            print("Move base cancel!")
            cancel_move_base(self.move_base_cancel)
            self.state = 0
        elif self.state == 9:
            print("End gravity compensation!")
            end_gravity_compensation(self.gravity_end)
            self.state = 0
    def start(self):
        self.listener.start()
    def run(self):
        # run the listener in non-blocking mode
        # self.listener.join()
        self.listener.join()


def main():
    # start_gravity_compensation(gravity_start)
    # cancel_move_base(move_base_cancel)
    # shutdown_python_programs()
    # input("Press Enter to continue...")

    return

if __name__ == '__main__':
    rospy.init_node('safety_run')
    # listener = PuckjsListener()
    # listener.start()
    # listener.run()
    listener_mouse = MouseClickListener()
    listener_mouse.run()

    # listener.run()
    # main()
