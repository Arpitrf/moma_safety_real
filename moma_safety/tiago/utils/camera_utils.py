import cv2
import imageio
import time

import numpy as np
import pyrealsense2 as rs
from typing import Dict
from threading import Thread
from datetime import datetime

class RealSenseCamera:
    def __init__(self, *args, **kwargs) -> None:
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        p = self.pipeline.start(config)
        depth_intr = p.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        rgb_intr = p.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._img_shape  = [rgb_intr.width, rgb_intr.height, 3]
        self._depth_shape = [depth_intr.width, depth_intr.height, 1]
        self.align = rs.align(rs.stream.color)

    def stop(self):
        self.pipeline.stop()

    def get_img(self) -> np.ndarray:
        return self.get_camera_obs()['image']

    def get_depth(self) -> np.ndarray:
        return self.get_camera_obs()['depth']

    def get_camera_obs(self) -> Dict[str, np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.expand_dims(np.asarray(depth_frame.get_data()), -1).astype(int) if depth_frame else None
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asarray(color_frame.get_data()).astype(int) if color_frame else None
        return {
            'image': color_image,
            'depth': depth_image,
        }

    @property
    def img_shape(self):
        return self._img_shape

    @property
    def depth_shape(self):
        return self._depth_shape


try:
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CameraInfo
    from moma_safety.tiago.utils.ros_utils import Listener

    def img_processing(data):
        br = CvBridge()
        img = cv2.cvtColor(br.imgmsg_to_cv2(data), cv2.COLOR_BGR2RGB)
        return np.array(img).astype(int)

    def depth_processing(data):
        br = CvBridge()
        img = br.imgmsg_to_cv2(data)
        return np.expand_dims(np.array(img), -1).astype(int)

    def flip_img(img):
        return np.flip(np.array(img).astype(int), axis=[0, 1])

    def uncompress_image(data):
        np_arr = np.fromstring(data.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return np.array(img).astype(int)

    def uncompress_depth(data):
        np_arr = np.fromstring(data.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        return np.expand_dims(np.array(img), -1).astype(int)

    # Handle when no depth is available
    class Camera:

        def __init__(self, img_topic, depth_topic, input_message_type=Image, camera_info_topic=None, img_post_proc_func=None, depth_post_proc_func=None, *args, **kwargs) -> None:
            self.img_topic = img_topic
            self.depth_topic = depth_topic

            self.img_listener = Listener(
                                input_topic_name=self.img_topic,
                                input_message_type=input_message_type,
                                post_process_func=img_processing if img_post_proc_func is None else img_post_proc_func
                            )
            self.depth_listener = Listener(
                                    input_topic_name=self.depth_topic,
                                    input_message_type=input_message_type,
                                    post_process_func=depth_processing if depth_post_proc_func is None else depth_post_proc_func
                                )

            self._img_shape  = self.img_listener.get_most_recent_msg().shape
            self._depth_shape = self.depth_listener.get_most_recent_msg().shape

            self.camera_info = None
            if camera_info_topic is not None:
                info_listener = Listener(
                                    input_topic_name=camera_info_topic,
                                    input_message_type=CameraInfo
                                )

                self.camera_info = info_listener.get_most_recent_msg()

        def get_img(self):
            return self.img_listener.get_most_recent_msg()

        def get_depth(self):
            return self.depth_listener.get_most_recent_msg()

        def get_camera_obs(self):
            return {
                'image': self.get_img(),
                'depth': self.get_depth(),
            }

        @property
        def img_shape(self):
            return self._img_shape

        @property
        def depth_shape(self):
            return self._depth_shape

        def stop(self):
            pass

except:
    pass

class RecordVideo:

    def __init__(self,
                 camera_interface_top=None,
                 camera_interface_side=None,
                 camera_interface_ego=None) -> None:
        self.recording = False
        self.env_video_frames = {}
        self.camera_interface_top = camera_interface_top
        self.camera_interface_side = camera_interface_side
        self.camera_interface_ego = camera_interface_ego
        if self.camera_interface_top is not None:
            self.env_video_frames['top'] = []
        if self.camera_interface_side is not None:
            self.env_video_frames['side'] = []
        if self.camera_interface_ego is not None:
            self.env_video_frames['ego'] = []

    def reset_frames(self):
        if self.camera_interface_top is not None:
            self.env_video_frames['top'] = []
        if self.camera_interface_side is not None:
            self.env_video_frames['side'] = []
        if self.camera_interface_ego is not None:
            self.env_video_frames['ego'] = []
    
    def setup_thread(self, target):
        print('SETUP THREAD', target)
        # print(list(map(lambda t:t.name,threading.enumerate())))
        thread = Thread(target=target)
        thread.daemon = True
        thread.start()
        print('started', thread.name)
        return thread

    def record_video_daemon_fn(self):
        counter = 0
        print("IN Daemon self.recording ", self.recording)
        while self.recorder_on:
            while self.recording:
                # if counter % 1000 == 0:
                time.sleep(0.1)
                if self.camera_interface_top is not None:
                    top_view = self.camera_interface_top.get_camera_obs()
                    capture_top = top_view["color"]
                    self.env_video_frames['top'].append(cv2.cvtColor(capture_top.copy(), cv2.COLOR_BGR2RGB))
                if self.camera_interface_side is not None:
                    side_view = self.camera_interface_side.get_camera_obs()
                    capture_side = side_view["color"]
                    self.env_video_frames['side'].append(cv2.cvtColor(capture_side.copy(), cv2.COLOR_BGR2RGB))
                if self.camera_interface_ego is not None:
                    #get_img
                    ego_view = self.camera_interface_ego.get_camera_obs()
                    capture_ego = ego_view["image"].astype(np.uint8)
                    # if capture_ego is not None:
                    #     if capture_ego.dtype == np.int32:  # Check if it's CV_32S
                    #         capture_ego = cv2.convertScaleAbs(capture_ego)  # Convert to CV_8U
                    self.env_video_frames['ego'].append(cv2.cvtColor(capture_ego.copy(), cv2.COLOR_BGR2RGB))
                # cv2.imshow("", cv2.cvtColor(capture.copy(), cv2.COLOR_BGR2RGB))
                # cv2.waitKey(10)
                # if counter % 100000 == 0:
                #     cv2.imwrite(f'temp/{counter}.jpg', top_view)
                # counter += 1
                # print("counter: ", counter)

    def setup_recording(self):
        if self.recording:
            return
        self.recorder_on = True
        self.recording_daemon = self.setup_thread(
            target=self.record_video_daemon_fn)

    def start_recording(self):
        self.recording = True
    
    def pause_recording(self):
        self.recording = False

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.recorder_on = False
        self.recording_daemon.join()
        self.recording_daemon = None

    def save_video(self, save_folder, epoch=None, traj_number=None):
        # print("self.env_video_frames.items(): ", self.env_video_frames.items())
        for key, frames in self.env_video_frames.items():
            if len(frames) == 0:
                continue
            print("len of frames: ", len(frames))

            current_time = datetime.now()
            f_name = current_time.strftime("%H-%M-%S")
            path = f'{save_folder}/{f_name}.mp4'
            if epoch is not None:
                path = f'{save_folder}/{epoch}_{traj_number}_{key}.mp4'
            with imageio.get_writer(path, mode='I', fps=10) as writer: # originally 24
                for frame in frames:
                    writer.append_data(frame)