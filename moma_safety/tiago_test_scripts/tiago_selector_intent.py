import os
import argparse
import pickle
import rospy
import numpy as np
from termcolor import colored
from easydict import EasyDict

import moma_safety.tiago.prompters.vlms as vlms
import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.utils.vision_utils as VU # vision_utils

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.skills.selector_intent import SkillIntentSelector
from moma_safety.tiago.skills import Reasoner
import moma_safety.tiago.RESET_POSES as RP
from moma_safety.tiago.skills import (
    MoveToSkill, PickupSkill, GoToLandmarkSkill, UseElevatorSkill, OpenDoorSkill, PushObsGrSkill,
    CallElevatorSkill, NavigateToPointSkill,
    # TurnSkill
)
from moma_safety.models.wrappers import GroundedSamWrapper
from moma_safety.models.wrapper_sam2 import GroundedSam2Wrapper

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time

import queue
import threading

from moma_safety.tiago.prompters.object_bbox import bbox_prompt_img

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
rospy.init_node('tiago_test')


reasoner = Reasoner()
def update_history(
        is_success,
        reason_for_failure,
        history_i,
        args,
    ):
    print(colored(f"Success: {is_success}", 'green' if is_success else 'red'))
    history_i['is_success'] = is_success

    # ask user if it is not successful
    success = U.confirm_user(True, 'Is the action successful (y/n)?')
    if success:
        history_i['is_success'] = True
    else:
        history_i['is_success'] = False
        if args.reasoner_type == 'oracle':
            U.clear_input_buffer()
            reason_for_failure = input('Reason for failure: ')
        elif args.reasoner_type == 'model':
            skill_info = history_i['skill_info']
            distance_info = history_i['distance_info']
            reason_for_failure, _ = reasoner.step(
                skill_name='selector',
                history_i=history_i,
                info={
                    'skill_info': skill_info,
                    'distance_info': distance_info,
                },
            )
            if type(reason_for_failure) == list:
                reason_for_failure = reason_for_failure[0]
        else:
            raise NotImplementedError
    history_i['model_analysis'] = reason_for_failure
    history_i['env_reasoning'] = None
    return history_i

def load_skill(skill_id, args, kwargs_to_add):
    if skill_id == 'move':
        prompt_args = {
            'raidus_per_pixel': 0.06,
            'arrow_length_per_pixel': 0.1,
            'plot_dist_factor': 1.0,
            'plot_direction': True,
        }
        skill = MoveToSkill(
            oracle_action=args.oracle,
            debug=False,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'pick_up_object':
        prompt_args = {
            'add_object_boundary': False,
            'add_arrows_for_path': False,
            'radius_per_pixel': 0.04,
        }
        skill = PickupSkill(
            oracle_position=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'goto_landmark':
        prompt_args = {
            'raidus_per_pixel': 0.03,
        }
        skill = GoToLandmarkSkill(
            bld=args.bld,
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'open_door':
        prompt_args = {}
        skill = OpenDoorSkill(
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'use_elevator':
        prompt_args = {
            'add_object_boundary': False,
            'add_dist_info': False,
            'add_arrows_for_path': False,
            'radius_per_pixel': 0.03,
        }
        skill = UseElevatorSkill(
            oracle_position=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'call_elevator':
        prompt_args = {
            'add_object_boundary': False,
            'add_dist_info': False,
            'add_arrows_for_path': False,
            'radius_per_pixel': 0.03,
        }
        skill = CallElevatorSkill(
            oracle_position=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'push_obs_gr':
        prompt_args = {}
        skill = PushObsGrSkill(
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'navigate_to_point_gr':
        prompt_args = {
            'raidus_per_pixel': 0.04,
            'arrow_length_per_pixel': 0.1, # don't need this
            'plot_dist_factor': 1.0, # don't need this
        }
        skill = NavigateToPointSkill(
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == "turn":
        prompt_args = {}
        skill = TurnSkill(
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    else:
        raise ValueError(f"Unknown skill id: {skill_id}")
    return skill


def get_kwargs_to_add(use_mini=False):
    print("Loading VLM")

    if use_mini:
        vlm = vlms.GPT4V(openai_api_key=os.environ['OPENAI_API_KEY'], model_name='gpt-4o-mini')
    else:
        vlm = vlms.GPT4V(openai_api_key=os.environ['OPENAI_API_KEY'])

    # test qwen
    # vlm = vlms.QWen(openai_api_key=None,
    #                 model_name='qwen2.5:7b',
    #                 )

    print("Done.")
    print("Loading transforms")
    if not args.no_robot:
        tf_map = TFTransformListener('/map')
        tf_odom = TFTransformListener('/odom')
        tf_base = TFTransformListener('/base_footprint')
        tf_arm_left = TFTransformListener('/arm_left_tool_link')
        tf_arm_right = TFTransformListener('/arm_right_tool_link')
        print("Done.")
        print("Loading action client for move_base")
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()
        print("Done")
        head_pub = Publisher('/head_controller/command', JointTrajectory)
        def process_head(message):
            return message.actual.positions
        head_sub = Listener('/head_controller/state', JointTrajectoryControllerState, post_process_func=process_head)
        kwargs_to_add = {
            'vlm': vlm,
            'tf_map': tf_map,
            'tf_odom': tf_odom,
            'tf_base': tf_base,
            'tf_arm_left': tf_arm_left,
            'tf_arm_right': tf_arm_right,
            'client': client,
            'head_pub': head_pub,
            'head_sub': head_sub,
            'skip_ros': args.no_robot,
        }
    else:
        kwargs_to_add = {
            'vlm': vlm,
            'skip_ros': args.no_robot,
        }
    return kwargs_to_add


def main(args):

    query = "test_intent"

    args.run_dir = args.run_dir + f'_{args.bld}_floor{args.floor_num}'
    if args.load_hist:
        args.run_dir += '_w_hist'
    args.run_dir += f'_{args.reasoner_type}'

    args.run_dir = os.path.join(args.run_dir, f'{query.replace(" ", "_")}')

    kwargs_to_add = get_kwargs_to_add(args.use_mini)
    kwargs_to_add['method'] = args.method

    dataset = [None]
    if args.data_dir:
        dataset = os.listdir(args.data_dir)
        dataset = [d for d in dataset if d.endswith('.pkl')]
        dataset = sorted(dataset)

    print("Loading gsam")
    gsam = None
    if args.run_vlm:
        # make sure the head is -0.8
        # gsam = GroundedSamWrapper(sam_ckpt_path=os.environ['SAM_CKPT_PATH'])
        gsam = GroundedSam2Wrapper()
    print("Gsam loading done")

    skill_id_list = args.skills
    skill_list = []
    for skill_id in skill_id_list:
        skill_list.append(load_skill(skill_id, args, kwargs_to_add=kwargs_to_add))
    skill_name2obj = {}
    skill_descs = []
    for skill in skill_list:
        skill.set_gsam(gsam)
        skill_name2obj[f'{skill.skill_name}'] = skill
        skill_descs.append(skill.skill_descs)

    if args.no_robot:
        env = None
    else:
        env = TiagoGym(
            frequency=10,
            right_arm_enabled=args.arm=='right',
            left_arm_enabled=args.arm=='left',
            right_gripper_type='robotiq2F-140' if args.arm=='right' else None,
            left_gripper_type='robotiq2F-85' if args.arm=='left' else None,
            base_enabled=False,
            torso_enabled=False,
        )

    prompt_args = {
        'n_vlm_evals': 0,
        'add_obj_ind': True,
        'raidus_per_pixel': 0.04,
        'add_dist_info': True,
        'add_object_boundary': False,
    }
    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)
    selector_skill = SkillIntentSelector(
        skill_descs=skill_descs,
        skill_names=skill_name2obj.keys(),
        run_dir=run_dir,
        prompt_args=prompt_args,
        add_histories=args.load_hist,
        reasoner_type=args.reasoner_type,
        **kwargs_to_add,
    )
    selector_skill.set_gsam(gsam)

    tf_listener = TFTransformListener('/base_footprint')
    grasp_h_r = RP.PICKUP_TABLE_L

    T_sub = 4
    eval_ind = 0
    floor_num = 2

    def create_video_from_images(images, video_name, fps=30):

        frame = images[0]
        height, width, layers = frame.shape

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4 video format
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        # Write each image into the video
        for image in images:
            video.write(image)

        # Release the video writer
        video.release()

    def display_video(data, rgb_lst):
        print(f"Displaying video: {data}")
        video_path = f'data/video_{data.split(".")[0]}.mp4'
        # if the video does not exist:
        if not os.path.exists(video_path):
            create_video_from_images(rgb_lst, video_path)

        # try:
        #     # Create a window
        #     window = tk.Tk()
        #     window.title("Video Player")

        #     # Open the video file
        #     cap = cv2.VideoCapture(video_path)

        #     # Create a label to display the frames
        #     label = tk.Label(window)
        #     label.pack()

        #     # Create a button to close the window
        #     close_button = tk.Button(window, text="Close", command=lambda: close_window(window))
        #     close_button.pack()

        #     def update_frame():
        #         # Capture frame-by-frame
        #         ret, frame = cap.read()

        #         if ret:
        #             # Convert the image from BGR (OpenCV format) to RGB
        #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #             # Convert the frame to an ImageTk format
        #             image = Image.fromarray(frame)
        #             imgtk = ImageTk.PhotoImage(image=image)

        #             # Update the label with the new frame
        #             label.imgtk = imgtk
        #             label.configure(image=imgtk)

        #             # Call the update_frame function again after 20ms
        #             label.after(20, update_frame)
        #         else:
        #             # Release the capture if the video has ended
        #             cap.release()

        #     # Start displaying the video
        #     update_frame()

        #     # Start the Tkinter event loop
        #     window.mainloop()

        #     time.sleep(2)

        #     window.destroy()

        # except:
        #     return


    def selector_worker(selector_skill, env, task, args):

        print("Starting selector_worker")

        # Unpack task details
        i, subsampled_history, obs_pp = task
        # NOTE: obs_pp is the obs at the very beginning (timestep i = 0)

        rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

        save_key = f'eval_id_{eval_ind:03d}_step_{select_ind:03d}'
        info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'eval_ind': eval_ind, 'save_key': save_key, 'floor_num': floor_num}

        # Call the selection_skill.step function
        is_success, selection_error, selection_history, selection_return_info = selector_skill.step(
            env=env,
            rgb=rgb,
            depth=depth,
            pcd=pcd,
            normals=normals,
            robot_history=subsampled_history,
            query="Predict what the human want to do next.",
            arm=args.arm,
            execute=args.exec,
            run_vlm=args.run_vlm,
            info=info,
            history=None,
            n_retries=1,
        )

        return is_success, selection_error, selection_history, selection_return_info


    def selector_worker_video(selector_skill, env, task, args):

        i, subsampled_history, obs_pp_lst = task

        rgb_lst = []

        for i in range(len(obs_pp_lst)):
            obs_pp = obs_pp_lst[i]

            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

            rgb_lst.append(rgb)

            if i == 0:
                save_key = f'eval_id_{eval_ind:03d}_step_{select_ind:03d}'
                info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'eval_ind': eval_ind, 'save_key': save_key, 'floor_num': floor_num}

        # Call the selection_skill.step function
        is_success, selection_error, selection_history, selection_return_info = selector_skill.step_video(
            env=env,
            rgb_lst=rgb_lst,
            depth=depth,
            pcd=pcd,
            normals=normals,
            robot_history=subsampled_history,
            query="Predict what the human want to do next.",
            arm=args.arm,
            execute=args.exec,
            run_vlm=args.run_vlm,
            info=info,
            history=None,
            n_retries=1,
        )

        return is_success, selection_error, selection_history, selection_return_info


    def selector_worker_video_mp(selector_skill, env, robot_history, tf_listener, task_queue, result_queue, skill_histories, stop_event, args):

        print("Starting selector_worker")
        # print(stop_event.is_set())

        while not stop_event.is_set():
            try:
                task = task_queue.get()
                # print("Task received in selector_worker: ", task)

                if task is None:
                    break

                i, subsampled_history, obs_pp_lst, encoded_image_lst, info = task

                rgb_lst = []

                for i in range(len(obs_pp_lst)):
                    obs_pp = obs_pp_lst[i]

                    rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

                    rgb_lst.append(rgb)

                    # if i == 0:
                        # save_key = f'eval_id_{eval_ind:03d}_step_{select_ind:03d}'
                        # info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'eval_ind': eval_ind, 'save_key': save_key, 'floor_num': floor_num}


                # Call the selection_skill.step function
                is_success, selection_error, selection_history, selection_return_info = selector_skill.step_video_subtask(
                    env=env,
                    encoded_image_lst=encoded_image_lst,
                    rgb_lst=rgb_lst,
                    depth=depth,
                    pcd=pcd,
                    normals=normals,
                    robot_history=subsampled_history,
                    query="Predict what the human want to do next.",
                    arm=args.arm,
                    execute=args.exec,
                    run_vlm=args.run_vlm,
                    info=info,
                    history=None,#skill_histories['selection'] if args.add_selection_history else None
                )

                # Put the results in the result queue for the main thread to read
                result_queue.put((is_success, selection_error, selection_history, selection_return_info))

            except queue.Empty:
                continue


    time_log_list = []

    # randomly permutate dataset
    np.random.shuffle(dataset)


    task_queue = queue.Queue()
    result_queue = queue.Queue()
    stop_event = threading.Event()

    if args.use_video:
        selector_thread = threading.Thread(target=selector_worker_video_mp, args=(selector_skill, env, None, tf_listener, task_queue, result_queue, None, stop_event, args))
    else:
        selector_thread = threading.Thread(target=selector_worker, args=(selector_skill, env, None, tf_listener, task_queue, result_queue, None, stop_event, args))

    selector_thread.start()

    for i, data in enumerate(dataset):

        if int(data.split('.')[0].split('_')[-1]) in [0,1]:
            continue

        if args.traj_num is not None:
            if int(data.split('.')[0].split('_')[-1]) != args.traj_num:
                continue

        # input("Press enter to continue")

        if data is not None:
            dataset_path = os.path.join(args.data_dir, data)
            robot_history, obs_pp_lst = pickle.load(open(dataset_path, 'rb'))
            robot_history = robot_history[:40]
            obs_pp_lst = obs_pp_lst[:40]

            # subsample trajectory for VLM input
            interval = len(robot_history) // T_sub
            subsampled_history = robot_history[::interval]
            subsampled_obs_pp_lst = obs_pp_lst[::interval]

            print(len(robot_history), len(subsampled_history), len(subsampled_obs_pp_lst))

        else:
            raise ValueError("No data provided")

        # playback video from the dataset
        rgb_lst = [U.convert_color(obs_pp['rgb']) for obs_pp in obs_pp_lst]

        display_video(data, rgb_lst)

        obs_pp = obs_pp_lst[0]
        rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

        select_ind = 0
        save_key = f'step_{select_ind:03d}'
        if data is not None:
            save_key = 'eval_' + data.split('.')[0]
        info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'save_key': save_key}
        info['floor_num'] = args.floor_num

        # if args.use_video:

        #     task = (i, subsampled_history, subsampled_obs_pp_lst)

        #     is_success, reason_for_failure, history_i, return_info = selector_worker_video(selector_skill, env, task, args)

        # else:

        #     task = (i, subsampled_history, obs_pp)

        #     is_success, reason_for_failure, history_i, return_info = selector_worker(selector_skill, env, task, args)


        rgb_lst = []

        for i in range(len(subsampled_obs_pp_lst)):
            obs_pp = obs_pp_lst[i]

            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

            rgb_lst.append(rgb)

            if i == 0:
                save_key = f'eval_id_{eval_ind:03d}_step_{select_ind:03d}'
                info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'eval_ind': eval_ind, 'save_key': save_key, 'floor_num': floor_num}

        encoded_image_lst = []
        for rgb_id in range(len(rgb_lst)):

            rgb = rgb_lst[rgb_id]

            im_copy_start = time.time()
            im = rgb.copy()
            im_copy_end = time.time()

            img_size = min(im.shape[0], im.shape[1])

            selector_skill.prompt_args.update({
                'radius': int(img_size * selector_skill.prompt_args['radius_per_pixel']),
                'fontsize': int(img_size * 30 * selector_skill.prompt_args['radius_per_pixel']),
            })
            info.update({'add_obj_ind': selector_skill.add_obj_ind})

            # Start the overall timing
            start_total = time.time()

            # only do this for the last image
            if selector_skill.add_obj_ind and rgb_id == len(rgb_lst) - 1:
                gsam_query = ['all objects']

                # Time the object bounding box retrieval
                start_bbox_retrieval = time.time()

                for _ in range(2):
                    bboxes, mask_image = selector_skill.get_object_bboxes(rgb, query=gsam_query)
                    if len(bboxes) > 0:
                        break
                    else:
                        gsam_query = ['all objects and floor']

                # End time for bbox retrieval
                end_bbox_retrieval = time.time()
                print(f"Bounding box retrieval took: {end_bbox_retrieval - start_bbox_retrieval:.4f} seconds")

                if len(bboxes) == 0:
                    # this should not happen
                    ipdb.set_trace()
                    error = "No objects found in the scene."
                    selector_skill.on_failure(
                        reason_for_failure=error,
                        reset_required=False,
                        capture_history={},
                        return_info={},
                    )

                # Time the overlay image creation
                start_overlay_creation = time.time()

                # used mainly for debugging
                overlay_image = U.overlay_xmem_mask_on_image(
                    rgb.copy(),
                    np.array(mask_image),
                    use_white_bg=False,
                    rgb_alpha=0.3
                )

                # End time for overlay creation
                end_overlay_creation = time.time()
                print(f"Overlay creation took: {end_overlay_creation - start_overlay_creation:.4f} seconds")

                # save the overlay image for debugging
                U.save_image(overlay_image, os.path.join(selector_skill.vis_dir, f'overlay_image_{info["save_key"]}_{rgb_id}.png'))

                # Time for bbox processing
                start_bbox_processing = time.time()

                bbox_id2dist = {}
                bbox_id2pos = {}
                for bbox in bboxes:
                    center = (bbox[1] + bbox[3]) // 2, (bbox[2] + bbox[4]) // 2
                    pos_wrt_base = pcd[center[1], center[0]]
                    dist = np.linalg.norm(pos_wrt_base[:2])
                    bbox_id2dist[bbox[0]] = dist
                    bbox_id2pos[bbox[0]] = pos_wrt_base
                    print(bbox[0], bbox_id2pos[bbox[0]])

                # End time for bbox processing
                end_bbox_processing = time.time()
                print(f"Bounding box processing took: {end_bbox_processing - start_bbox_processing:.4f} seconds")

                info.update({
                    'bbox_ignore_ids': [0],
                    'bbox_id2dist': bbox_id2dist,
                    'bbox_id2pos': bbox_id2pos,
                })

                # Time for prompt image creation
                start_prompt_img = time.time()

                prompt_rgb, obj_bbox_list = bbox_prompt_img(
                    im=rgb.copy(),
                    info=info,
                    bboxes=bboxes,
                    prompt_args=selector_skill.prompt_args,
                )

                # End time for prompt image creation
                end_prompt_img = time.time()
                print(f"Prompt image creation took: {end_prompt_img - start_prompt_img:.4f} seconds")

                info['obj_bbox_list'] = obj_bbox_list
                U.save_image(prompt_rgb, os.path.join(selector_skill.vis_dir, f'prompt_img_{info["save_key"]}_{rgb_id}.png'))

            else:
                im_copy_start = time.time()

                prompt_rgb = rgb.copy()

                im_copy_end = time.time()

            start_encoding = time.time()

            encoded_image = U.encode_image(prompt_rgb)
            encoded_image_lst.append(encoded_image)

            end_encoding = time.time()

            print(f"Encoding took: {end_encoding - start_encoding:.4f} seconds")

            # End the overall timing
            end_total = time.time()
            print(f"Total execution time: {end_total - start_total:.4f} seconds")


        if args.use_video:
            task_queue.put((i, subsampled_history, subsampled_obs_pp_lst, encoded_image_lst, info))
        else:
            task_queue.put((i, subsampled_history, obs_pp))

        while True:
            if not result_queue.empty():
                is_success, error, history, return_info = result_queue.get()
                break

        time_info = return_info['time']
        time_log_list.append(time_info)
        # save time log
        time_log_path = os.path.join(args.run_dir, f'time_log_video_{args.use_video}_gptmini_{args.use_mini}_random.pkl')
        pickle.dump(time_log_list, open(time_log_path, 'wb'))
        # calculate average time

        # for time in time_log_list:
        #     print(time)

        # # calculate the average for each of the time_log in time_log_lst
        # time_copy_log, time_gsam_log, time_prompts_log, time_subtask_log, time_vlm_log, time_end_log, time_overall_log = [np.mean([time_log_list[i][j] for i in range(len(time_log_list))]) for j in range(len(time_log_list[0]))]
        # print(f"\033[1;32;40mAverage time log:\033[0m")
        # print(f"\033[1;32;40mTime for copying: {time_copy_log:.2f} seconds\033[0m")
        # print(f"\033[1;32;40mTime for GSAM: {time_gsam_log:.2f} seconds\033[0m")
        # print(f"\033[1;32;40mTime for prompts: {time_prompts_log:.2f} seconds\033[0m")
        # print(f"\033[1;32;40mTime for subtask: {time_subtask_log:.2f} seconds\033[0m")
        # print(f"\033[1;32;40mTime for VLM: {time_vlm_log:.2f} seconds\033[0m")
        # print(f"\033[1;32;40mTime for end: {time_end_log:.2f} seconds\033[0m")
        # print(f"\033[1;32;40mTime for overall: {time_overall_log:.2f} seconds\033[0m")


        skill_name = return_info['skill_name']
        subtask = return_info['subtask']
        common_args = {
            'env': env,
            'rgb': rgb,
            'depth': depth,
            'pcd': pcd,
            'normals': normals,
            'arm': args.arm,
            'info': info,
            'history': None,
            'query': subtask, # or task_query
        }
        if skill_name == 'goto_landmark':
            goto_is_success, goto_reason_for_failure, goto_history, goto_return_info = \
                    skill_name2obj[skill_name].step(
                        **common_args,
                        execute=False,
                        run_vlm=args.run_vlm,
                        debug=args.debug,
                        floor_num=args.floor_num,
                        bld=args.bld,
                    )
        if args.save_hist:
            save_key = info['save_key']
            eval_dir = args.run_dir
            save_dir = eval_dir
            print(save_key) # adjust the save_key before saving the pkl file
            history_path = os.path.join(save_dir, f'history_{save_key}.pkl')
            history_i = update_history(
                is_success,
                reason_for_failure,
                history_i,
                args=args,
            )
            pickle.dump(history_i, open(history_path, 'wb'))

    rospy.signal_shutdown("Shutdown")
    rospy.spin()


if __name__ == '__main__':
    # ROS_HOSTNAME=localhost ROS_MASTER_URI=http://localhost:11311 python tiago_test_scripts/tiago_selector.py --run_vlm --prev-traj-pkl /home/abba/Desktop/rutav/vlm-skill/datasets/move_009.pkl --llm-baseline
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='../datasets/selector_test')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exec', action='store_true')
    parser.add_argument('--arm', type=str, default='left', choices=['right', 'left'])
    parser.add_argument('--run_vlm', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--floor_num', type=int, help='starting floor number', default=2)
    parser.add_argument('--load_hist', action='store_true')
    parser.add_argument('--save_hist', action='store_true')
    parser.add_argument('--reasoner_type', type=str, default='oracle', choices=['oracle', 'model'])
    parser.add_argument('--prev-traj-pkl', type=str, default="")  # pkl file to load previous env readings from
    parser.add_argument(
        '--method', default="ours",
        choices=["ours", "llm_baseline", "ours_no_markers"])
    parser.add_argument('--bld', default='ahg', choices=['ahg', 'mbb', 'nhb'])
    parser.add_argument('--traj-num', type=int, default=None)
    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--use_mini', action='store_true')

    args = parser.parse_args()
    # args.skills = ['pick_up_object', 'move', 'goto_landmark', 'open_door', 'push_obs_gr']
    args.skills = ['pick_up_object', 'navigate_to_point_gr', 'open_door']
    # args.skills = ['pick_up_object', 'place_object']

    assert args.run_vlm or args.oracle
    assert args.run_dir is not None

    if (args.prev_traj_pkl) or (args.data_dir):
        args.no_robot = True
        assert args.exec == False

    main(args)
