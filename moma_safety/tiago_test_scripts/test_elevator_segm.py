import os
import pickle
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from termcolor import colored
from easydict import EasyDict

import moma_safety.utils.utils as U
from moma_safety.tiago.skills import Reasoner
import moma_safety.tiago.prompters.vip_utils as vip_utils
import moma_safety.tiago.prompters.vlms as vlms # GPT4V
from moma_safety.models.wrappers import GroundedSamWrapper
from moma_safety.tiago.prompters.object_bbox import bbox_prompt_img
from moma_safety.tiago.skills.use_elevator import get_button_positions, make_prompt, make_prompt_floor_ch, UseElevatorSkill, CallElevatorSkill, make_history_call_elevator, make_history_use_elevator

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def create_history_msgs(
        vlm,
        history,
        func, # function to create the prompt
        func_kwargs, # kwargs for the function
    ):
    history_msgs = []
    history_inst, history_desc, history_model_analysis = func(history, **func_kwargs)
    history_imgs = []
    for msg in history:
        assert 'image' in msg
        # check if the image is already encoded
        assert isinstance(msg['image'], np.ndarray), f"Image is not a numpy array, but {type(msg['image'])}"
        # encode the image. First convert to bgr and then encode.
        encoded_image = U.encode_image(msg['image'])
        history_imgs.append(encoded_image)

    history_msgs = vlm.create_msg_history(
        history_instruction=history_inst,
        history_desc=history_desc,
        history_model_analysis=history_model_analysis,
        history_imgs=history_imgs,
    )
    return history_msgs

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
    reason_for_failure = None
    if success:
        history_i['is_success'] = True
    else:
        history_i['is_success'] = False
        if args.reasoner_type == 'oracle':
            U.clear_input_buffer()
            reason_for_failure = input('Reason for failure: ')
        elif args.reasoner_type == 'model':
            reason_for_failure, _ = reasoner.step(
                skill_name=args.skill_type,
                history_i=history_i,
                info={'floor_num': args.floor_num},
            )
            if type(reason_for_failure) == list:
                reason_for_failure = reason_for_failure[0]
        else:
            raise NotImplementedError
        # import ipdb; ipdb.set_trace()
        # input('Press any key to continue...')
    history_i['model_analysis'] = reason_for_failure
    history_i['env_reasoning'] = None
    return history_i

def get_param_from_response(response):
    '''
        skill_specific function to get the param from the vlm response
    '''
    return_info = {}
    return_info['response'] = response
    return_info['error_list'] = []
    object_id = ''
    try:
        object_id = vip_utils.extract_json(response, 'button_id')
        print(f"Buton ID: {object_id}")
    except Exception as e:
        print(str(e))
        object_id = ''
    return_info['button_id'] = object_id
    return object_id, return_info

def get_param_from_response_floor_num(response, return_info):
    '''
        skill_specific function to get the param from the vlm response
    '''
    floor_num = ''
    try:
        floor_num = vip_utils.extract_json(response, 'target_floor_num')
        print(f"Floor number: {floor_num}")
    except Exception as e:
        print(str(e))
        floor_num = ''
    return_info['floor_num'] = floor_num
    return_info['target_floor_num'] = floor_num
    return floor_num, return_info

def prep_llm_prompt(
        vlm,
        encoded_image,
        make_prompt_func,
        make_prompt_func_kwargs):
    def scene_prompt_func():
        """Used for baseline to provide textual description of image"""
        instructions = """
INSTRUCTIONS:
You will be given an image of the scene. First, describe the scene in the image. Then, describe each marked object briefly.
Provide all the descriptions at the end in a valid JSON of this format: {{"scene_description": "", "obj_descriptions", ""}}"""
        task_prompt = """
ANSWER: Let's think step-by-step."""
        return instructions, task_prompt
    def get_param_from_scene_obj_resp(response):
        error_list = []
        return_info = {}
        return_info['response'] = response

        scene_desc = ''
        try:
            scene_desc = vip_utils.extract_json(response, 'scene_description')
        except Exception as e:
            print(f"Error: {e}")
            error = 'Missing scene description information in the JSON response.'
            error_list.append(error)

        obj_descs = ''
        try:
            obj_descs = vip_utils.extract_json(response, 'obj_descriptions')
        except Exception as e:
            print(f"Error: {e}")
            obj_descs = None
            error = 'Missing skill name in the JSON response.'
            error_list.append(error)
        if isinstance(obj_descs, dict):
            obj_id2desc_map = dict(obj_descs)
            obj_descs = ""
            for _id in sorted(obj_id2desc_map.keys()):
                obj_descs += f"{_id}: {obj_id2desc_map[_id]} "
            obj_descs = obj_descs.strip()

        return_info['error_list'] = error_list
        return_info['scene_desc'] = scene_desc
        return_info['obj_descs'] = obj_descs
        return scene_desc, obj_descs, return_info
    # First get a textual description of the scene and object IDs from image.
    instructions, task_prompt = scene_prompt_func()
    prompt_seq = [task_prompt, encoded_image]
    scene_obj_desc_response = vlm.query(instructions, prompt_seq)
    scene_desc, obj_descs, scene_return_info = (
        get_param_from_scene_obj_resp(scene_obj_desc_response))
    # print(colored("Scene Desc:\n" + scene_desc, 'blue'))
    # print(colored("Obj descs:\n" + obj_descs, 'blue'))

    # Update make_prompt_func_kwargs for actual selector/skill call
    llm_baseline_prompt_info = dict(
        im_scene_desc=scene_desc,
        obj_descs=obj_descs,
    )
    make_prompt_func_kwargs.update(dict(
        llm_baseline_info=llm_baseline_prompt_info,
        method=method,
    ))
    instructions, task_prompt = make_prompt_func(**make_prompt_func_kwargs)
    prompt_seq = [task_prompt]
    return instructions, prompt_seq
def vlm_runner(
    vlm,
    encoded_image,
    history_msgs,
    make_prompt_func,
    make_prompt_func_kwargs,
    force_vlm_prompt=False,
    method='ours',
):
    if method == "llm_baseline" and not force_vlm_prompt:
        instructions, prompt_seq = prep_llm_prompt(
            vlm,
            encoded_image,
            make_prompt_func,
            make_prompt_func_kwargs)
        task_prompt = "".join(prompt_seq)
    else:
        make_prompt_func_kwargs.update(dict(method=method))
        instructions, task_prompt = make_prompt_func(**make_prompt_func_kwargs)
        prompt_seq = [task_prompt, encoded_image]
    if method != "ours":
        print(colored(f"{method} Prompt\n" + instructions + task_prompt, 'light_blue'))
    response = vlm.query(instructions, prompt_seq, history=history_msgs)
    print(f"*******************************************************")
    print(colored(response, 'yellow'))
    return response

prompt_args = {
    "color": (0, 0, 0),
    "mix_alpha": 0.6,
    'thickness': 2,
    'rgb_scale': 255,
    'add_object_boundary': False,
    'add_dist_info': False, # not used in this function
    'add_dist_info': False,
    'add_arrows_for_path': False,
    'path_start_pt': (0, 0),
    'path_end_pt': (0, 0),
    'radius_per_pixel': 0.03,
    'plot_outside_bbox': True,
}
floor_num = 2
query = None
if floor_num == 1:
    query=f"Go to the second floor"
elif floor_num == 2:
    # query="Call the elevator to go to the first floor."
    query=f"Go to the first floor"
save_history = False
load_history = True
radius_per_pixel = prompt_args['radius_per_pixel']
gsam = GroundedSamWrapper(sam_ckpt_path=os.environ["SAM_CKPT_PATH"])

# call_elevator:
# with model history: 20/20, 10/10 (new building)
# with oracle history: 20/20, 9/10 (new building)
# with no history: 17/20, 9/10 (new building)

# use elevator
# with model history: 10/10, 10/10 (going up)
# with oracle history: 10/10, 10/10 (going up)
# without history: 8/10, 10/10 (going up)
reasoner = Reasoner()
# model_name = 'claude-3-5-sonnet-20240620'
# model_name = 'claude-3-opus-20240229'
# model_name = 'claude-3-sonnet-20240229'
# model_name = 'claude-3-haiku-20240307'
model_name = 'gpt-4o-mini-2024-07-18'

method = 'ours'
skill_type = 'call_elevator'
reasoner_type = 'model'
dataset_name = None
if skill_type == 'use_elevator':
    dataset_name = 'use_elevator_ahg_in'
elif skill_type == 'call_elevator':
    # dataset_name = 'test_elevator_mbb_north_out2'
    # dataset_name = 'call_elevator_ahg2'
    dataset_name = 'call_elevator_mbb'
    # dataset_name = 'call_elevator_ahg'
# base_dir_name = f'prompt_data_elev_floor{floor_num}_{reasoner_type}'
base_dir_name = f'{method}_floor_num{floor_num}_hist{load_history}_{model_name}'
data_dir = f"/home/pal/Desktop/rutav/vlm-skill/../datasets/final_ablations/{dataset_name}"

eval_dir = base_dir_name
# eval_dir = base_dir_name if not load_history else base_dir_name + '_w_hist'
# eval_dir = eval_dir + '_testv2'
save_dir = os.path.join(data_dir, eval_dir)
os.makedirs(save_dir, exist_ok=True)
imgs = os.listdir(data_dir)
imgs = [img for img in imgs if img.endswith('.png')]
imgs = sorted(imgs)
num_data = len(imgs)

#### TESTING CALL ELEVATOR SKILL
# use_elevator = CallElevatorSkill(
#     oracle_position=False,
#     debug=False,
#     run_dir=save_dir,
#     prompt_args=prompt_args,
#     skip_ros=True,
#     add_histories=True,
# )
####
vlm = U.get_model(model_name)
history_msgs = None
history_list = []
if load_history:
    base_dir, history_eval_dirs = None, None
    if skill_type == 'use_elevator':
        base_dir = "/home/pal/Desktop/rutav/datasets/use_elevator_ahg_in/prompt_data_elev_floor2_model/"
        history_eval_dirs = [os.path.join(base_dir, 'eval_id001.pkl')]
    elif skill_type == 'call_elevator':
        # base_dir = '/home/pal/Desktop/rutav/datasets/test_elevator_mbb_north_out2/prompt_data_elev/'
        # history_eval_dirs = [os.path.join(base_dir, 'eval_id003.pkl'), os.path.join(base_dir, 'eval_id011.pkl')]
        # history_eval_dirs = [os.path.join(base_dir, 'eval_id003.pkl')]

        # base_dir = '/home/pal/Desktop/rutav/datasets/test_elevator_mbb_north_out2/prompt_data_elev_floor1/'
        # eval_id000: The button id A has a red circle around the button indicating that it must only be pressed for emergency service.
        # eval_id005: The robot wants to go up from the first floor to the second floor. To go to a floor higher than the current floor, the up button (one which is above) must be pressed.
        # history_eval_dirs = [os.path.join(base_dir, 'eval_id000.pkl'), os.path.join(base_dir, 'eval_id005.pkl')]
        base_dir = '/home/pal/Desktop/rutav/datasets/test_elevator_mbb_north_out2/prompt_data_elev_floor1_model/'
        history_eval_dirs = [os.path.join(base_dir, 'eval_id000.pkl')]
    else:
        raise NotImplementedError
    for hist_eval_dir in history_eval_dirs:
        samples_per_hist = 1
        _history_all_path = os.path.join(hist_eval_dir, 'history_all.pkl')
        if hist_eval_dir.endswith('.pkl'):
            _history_all_path = hist_eval_dir
        assert os.path.exists(_history_all_path), f"History file not found: {_history_all_path}"
        _history_list = pickle.load(open(_history_all_path, 'rb'))
        if not isinstance(_history_list, list):
            _history_list = [_history_list]
        # _success_list = [h for h in _history_list if h['is_success']]
    for hist_eval_dir in history_eval_dirs:
        samples_per_hist = 1
        _history_all_path = os.path.join(hist_eval_dir, 'history_all.pkl')
        if hist_eval_dir.endswith('.pkl'):
            _history_all_path = hist_eval_dir
        assert os.path.exists(_history_all_path), f"History file not found: {_history_all_path}"
        _history_list = pickle.load(open(_history_all_path, 'rb'))
        if not isinstance(_history_list, list):
            _history_list = [_history_list]
        # _success_list = [h for h in _history_list if h['is_success']]
        _history_list = [h for h in _history_list if not h['is_success']]
        _history_list = _history_list[:samples_per_hist]
        # _success_list = _success_list[:samples_per_hist]
        history_list.extend(_history_list)

    history_msgs = create_history_msgs(
        vlm=vlm,
        history=history_list,
        func=make_history_call_elevator if skill_type == 'call_elevator' else make_history_use_elevator,
        func_kwargs={},
    )

for img_ind, img in enumerate(imgs):
    url = os.path.join(data_dir, img)
    save_url = os.path.join(save_dir, img)
    os.makedirs(save_dir, exist_ok=True)
    text = "buttons"
    image = np.array(Image.open(url))

    final_mask_image = gsam.segment(image, [text], filter_threshold=200)
    print(np.unique(final_mask_image))
    if len(np.unique(final_mask_image)) == 0:
        print(f"Could not find {text} in {img}")
        continue
    overlay_image = U.overlay_xmem_mask_on_image(
        image.copy(),
        np.array(final_mask_image),
        use_white_bg=False,
        rgb_alpha=0.3
    )

    img_file = os.path.join(save_dir, f'overlay_{img}.png')
    Image.fromarray(overlay_image).save(img_file)

    img_size = min(image.shape[:2])
    prompt_args.update({
        'radius': int(img_size * radius_per_pixel),
        'fontsize': int(img_size * 30 * radius_per_pixel),
    })
    bboxes, final_mask_image = get_button_positions(image, np.asarray(final_mask_image))

    bbox_id2dist = {}
    for bbox in bboxes:
        bbox_id = bbox[0]
        bbox_id2dist[bbox_id] = 0.0
    info = {
        'bbox_ignore_ids': [0],
        'bbox_id2dist': bbox_id2dist,
        'save_key': f'out_{img}',
        'floor_num': floor_num,
    }

    prompt_rgb = None
    if method == 'ours_no_markers':
        prompt_rgb = image.copy()
    else:
        prompt_rgb, obj_bbox_list = bbox_prompt_img(image, bboxes, info=info, prompt_args=prompt_args)

    Image.fromarray(prompt_rgb).save(save_url)
    encoded_image = U.encode_image(prompt_rgb)
    response = vlm_runner(
        vlm=vlm,
        encoded_image=encoded_image,
        history_msgs=history_msgs,
        make_prompt_func=make_prompt if skill_type == 'call_elevator' else make_prompt_floor_ch,
        make_prompt_func_kwargs={
            'query': query,
            'info': info,
            'method': method,
        },
        method=method,
    )
    button_id, return_info = get_param_from_response(response)
    if skill_type == 'use_elevator':
        floor_num, return_info = get_param_from_response_floor_num(response, return_info)
    print("BUTTON ID: ", button_id)
    capture_history = {
        'image': prompt_rgb,
        'query': query,
        'model_response': [button_id, floor_num],

        'full_response': response,
        'button_id': button_id,
        'floor_num': floor_num,
        'model_analysis': '',
    }
    print(capture_history['model_response'])

    img_file = os.path.join(save_dir, f'output_{info["save_key"]}.png')
    U.save_model_output(
        rgb=prompt_rgb,
        response=response,
        subtitles=[],
        img_file=img_file,
    )
    save_key = info['save_key']
    history_path = os.path.join(save_dir, f'history_{save_key[:-4]}.pkl')
    args = EasyDict({
        'reasoner_type': reasoner_type,
        'skill_type': skill_type,
        'floor_num': floor_num,
    })
    if save_history:
        history_i = update_history(
            False,
            "",
            capture_history,
            args=args,
        )
        pickle.dump(history_i, open(history_path, 'wb'))
