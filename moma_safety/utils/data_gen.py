import os
import cv2
import math
import copy
import numpy as np

from termcolor import colored
import seaborn as sns
from vlm_skill.utils import utils as U
from vlm_skill.prompters.base import PromptBase

class ImgGrid:
    def __init__(
            self,
            grid_center: tuple[int, int],
            grid_size: tuple[int, int],
            grid_color: tuple[int, int, int] = (255, 255, 255),
    ):
        self.grid_size = grid_size
        self.grid_color = grid_color
        self.num_label_size = num_label_size

    @property
    def center(self):
        return self.grid_center

class ImgSegment:
    def __init__(
            self,
            unq_id: int,
            mask: np.ndarray,
            center: tuple[int, int],
            area: int,
            segm_id: int
        ):
        self.unq_id = unq_id
        self.mask = mask
        self.center = center
        self.area = area
        self.segm_id = segm_id
        self.contours, self.hierarchy = self.get_contour()
        self.sample_pts = {} # point, label

    def get_contour(self):
        '''
            Given a mask, get the contour and hierarchy.
            Mask is boolean array.
        '''
        mask = self.mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # assert len(contours) == 1, f"Number of contours {len(self.contours)} not equal to 1"
        contours = np.array([point for contour in contours for point in contour])

        contours = np.squeeze(contours, axis=1).tolist()
        contours = [(contour[1], contour[0]) for contour in contours]
        return contours, hierarchy

    def sample_points(self, n_points: int, _type: str, mode: str) -> list[tuple[int, int]]:
        '''
            Given a mask, sample n_points from the contour/mask.
        '''
        # import ipdb; ipdb.set_trace()
        points = None
        if _type == 'contour':
            points = self.contours
        elif _type == 'mask':
            points = np.where(self.mask)
            points = [(points[0][i], points[1][i]) for i in range(len(points[0]))]
        else:
            raise ValueError(f"Sampling Type {_type} not recognized")
        n_points = min(n_points, len(points))

        pts = []
        if mode == 'random':
            idx = np.random.choice(len(points), n_points, replace=False)
            pts = [(points[i][0], points[i][1]) for i in idx]
        elif mode == 'uniform':
            idx = np.linspace(0, len(points)-1, n_points, dtype=int)
            pts = [(points[i][0], points[i][1]) for i in idx]
        elif mode == 'farthest':
            # sample the first point randomly
            idx = np.random.choice(len(points), 1, replace=False)[0]
            pts.append(points[idx])
            # for the rest of the points
            for i in range(n_points-1):
                pts_np = np.array(pts)
                points_np = np.array(points)
                dist = np.sum((pts_np[:, np.newaxis] - points_np) ** 2, axis=2)
                # maximize the minimum distance
                dist = np.min(dist, axis=0) # sum of distance from all the sampled points
                idx = np.argmax(dist)
                pts.append(points[idx])
        return pts

class DataGenerator:
    def __init__(self):
        pass
    def generate_obs(self, obs: dict):
        raise NotImplementedError
    def generate_data(self, obs: dict):
        raise NotImplementedError

class GridOverlayNum(DataGenerator):
    def __init__(
        self,
        # grid parameters
        grid_size_per_pixel: tuple[int, int] = 0.2,
        center_label_type: str = 'num',
    ):
        self.grid_size_per_pixel = grid_size_per_pixel
        self.center_label_type = center_label_type
        super().__init__()

    def generate_obs(
            self,
            obs: dict,
            views: list[str],
            ooi: list[str] = ["all objects"],
        ):
        '''
            Given an observation in dict,
            - divide the image into different grids.
            - add black grid lines to the image.
            - add labels to the center of the grid.
        '''
        raise NotImplementedError
        return obs

class SegmentOverlayNum(DataGenerator):
    def __init__(
            self,
            # image processing
            mixing_alpha: float = 0.2,
            max_objects: int = 10,
            overlay_img: bool = True,

            # Sampling points in the segment
            sample_type: str = 'contour',
            sample_mode: str = 'random',
            sample_points_per_area: float = 1/3000,
            # For each point
            add_circle_at_point: bool = False,
            circle_for_point_radius: int = 8,
            circle_for_point_color: tuple[int, int, int] = (128, 128, 128),

            # Label for the center of the segment
            center_label_type: str = 'num',
            font_type: int = cv2.FONT_HERSHEY_SIMPLEX,
            font_scale: float = 0.4,
            font_color: tuple[int, int, int] = (255, 255, 255),
            font_thickness: int = 1,
            line_type: int = cv2.LINE_AA,
            add_circle_around_label: bool = True,
            # label_circle_radius: int = 8, # this is scaled w.r.t font_scale
            # segmentation of object function
            segment_img: bool = True,
            segm_obj_model: str = 'gsam',
            sam_ckpt_path: str = "sam_vit_h_4b8939.pth",
            label_circle_color: tuple[int, int, int] = (128, 128, 128),
            *args,
            **kwargs
        ):
        label_circle_radius = math.ceil((8*font_scale)/0.45)
        self.mixing_alpha = mixing_alpha
        self.max_objects = max_objects
        self.overlay_img = overlay_img

        # Label for the center of the segment
        self.center_label_type = center_label_type
        self.font_type = font_type
        self.font_scale = font_scale
        self.font_color = font_color
        self.font_thickness = font_thickness
        self.line_type = line_type
        self.label_circle_radius = label_circle_radius
        self.label_circle_color = label_circle_color
        self.add_circle_around_label = add_circle_around_label

        # Sampling points in the segment
        self.sample_type = sample_type
        self.sample_mode = sample_mode
        self.sample_points_per_area = sample_points_per_area
        # For each point in the segment
        self.add_circle_at_point = add_circle_at_point
        self.circle_for_point_radius = circle_for_point_radius
        self.circle_for_point_color = circle_for_point_color

        # segmentation of object function
        self.segment_img = segment_img
        self.segm_obj_model = segm_obj_model
        if self.segment_img:
            if self.segm_obj_model == 'gsam':
                from vlm_skill.models.wrappers import GroundedSamWrapper
                self.gsam = GroundedSamWrapper(sam_ckpt_path=sam_ckpt_path)

        self.unq_id2color = {}
        # use a color palette from a package with bright colors
        palette = sns.color_palette("bright", max_objects)
        # convert the color palette to 255 scale and store it in unq_id2color
        for i, color in enumerate(palette):
            assert len(color) == 3, f"Color {color} is not in RGB format"
            self.unq_id2color[i] = np.asarray([int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)])

        super().__init__(*args, **kwargs)

    def unique_segs(self, segm: np.ndarray, ignore_ids: list[int] = [0]):
        '''
            Given a segm image,
            count the number of unique segments.
        '''
        unique_segs = np.unique(segm)
        unique_segs = [seg for seg in unique_segs if seg not in ignore_ids]
        return len(unique_segs), unique_segs

    def get_segm_objs(self, segm: np.ndarray, unique_segm_ids: list[int]):
        '''
            Given a segm image and unique_segm_ids,
            create a mask, unique_segment_mask, center of segment, area of segment.
        '''
        segmented_objects = []
        unq_id = 1
        for seg_id in unique_segm_ids:
            mask = segm == seg_id
            center = np.mean(np.where(mask), axis=1)
            area = np.sum(mask)
            print(colored(f"Segment {seg_id} has area {area}", 'blue'))
            segmented_objects.append(ImgSegment(
                unq_id=unq_id,
                mask=mask,
                center=center,
                area=area,
                segm_id=seg_id
            ))
            unq_id += 1
        return segmented_objects

    def overlay_segs(self, rgb: np.ndarray, segs: list[ImgSegment]):
        '''
            Given an rgb image and a segments,
             - sort the segments by area
             - merge the masks of the segments and call it the final mask image

        '''
        # sort the segments by area
        segs = sorted(segs, key=lambda x: x.area, reverse=True)
        final_mask_image = np.zeros_like(segs[0].mask, dtype=np.uint8)

        for seg in segs:
            final_mask_image = np.where(seg.mask, seg.unq_id, final_mask_image)

        rgb = U.overlay_xmem_mask_on_image(
            rgb,
            final_mask_image,
            use_white_bg=False,
            rgb_alpha=self.mixing_alpha
        )
        return rgb

    def plot_points_on_image(
            self,
            rgb: np.ndarray,
            seg: ImgSegment,
            points: list[tuple[int, int]],
            center_label_type: str = 'num',
        ):
        # import ipdb; ipdb.set_trace()
        if center_label_type == 'num':
            center_label = str(seg.unq_id)
        elif center_label_type == 'char':
            center_label = chr(65 + seg.unq_id)
        else:
            raise ValueError(f"center_label_type {center_label_type} not recognized")
        # convert rgb to bgr
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for idx,point in enumerate(points):
            # add a circle at the center
            if self.add_circle_around_label:
                rgb = cv2.circle(rgb, (int(point[1]), int(point[0])), self.label_circle_radius, self.label_circle_color, -1)
            if self.add_circle_at_point:
                rgb = cv2.circle(rgb, (int(point[1]), int(point[0])), self.circle_for_point_radius+1, (0,0,0), -1)
                rgb = cv2.circle(rgb, (int(point[1]), int(point[0])), self.circle_for_point_radius, self.circle_for_point_color, -1)
            l = center_label + str(idx+1)
            seg.sample_pts[l] = point
            # add the center label to the image
            # adjust the center such that the text must inside the circle and not outside
            # point = (point[0] + int((5*self.font_scale)/0.4), point[1] - int((5*self.font_scale)/0.4))
            rgb = cv2.putText(rgb, l, (int(point[1]), int(point[0])), self.font_type, self.font_scale, self.font_color, self.font_thickness, self.line_type)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb, seg

    def add_label(
            self,
            rgb: np.ndarray,
            seg: ImgSegment,
            # Sampling points in the segment
            n_sample_points: int = None,
            sample_type: str = None,
            mode: str = None,
            position: str = 'center',
            center_label_type: str = 'num',
        ):
        '''
            Given an rgb image and a segment,
            add the center label to the image
        '''
        if n_sample_points is None:
            # n_sample_points = int(2*seg.area/100)
            n_sample_points = max(0, int(seg.area * self.sample_points_per_area))
            n_sample_points = min(n_sample_points, 5)
        if sample_type is None:
            sample_type = self.sample_type
        if mode is None:
            mode = self.sample_mode
        print(colored(f"Sampling {n_sample_points} points from the segment {seg.segm_id}", 'blue'))
        assert position == 'center', f"position {position} not recognized."
        points = [seg.center]
        if n_sample_points > 0:
            points.extend(
                seg.sample_points(
                    n_sample_points,
                    _type=sample_type,
                    mode=mode
                )
            )
        rgb, seg = self.plot_points_on_image(
            rgb=rgb,
            seg=seg,
            points=points,
            center_label_type=center_label_type
        )
        return rgb, seg

    def generate_obs(
            self,
            obs: dict,
            views: list[str],
            ooi: list[str] = ["all objects"],
        ):
        '''
            Given an observation in dict,
            - Take the segm image and count the number of unique segments.
            - For each unique segment,
                create a mask
                unique_segment_mask
                center of segment
                area of segment
            - overlay it on the rgb image.
            - Add the overlayed rgb image to the observation with key o_rgb.
        '''
        num_segs = {}
        obs['o_rgb'] = {}
        obs['segm_objs'] = {}
        for view in views:
            obs['o_rgb'][view] = None
            obs['segm_objs'][view] = None
            if self.segment_img and (obs['segm'][view] is None):
                assert self.segment_img, f"Segmentation image is None, but segment_img is set to {self.segment_img}"
                obs['segm'][view] = np.asarray(self.gsam.segment(obs['rgb'][view], ooi))

            segm = obs['segm'][view]
            num_segs, unique_segm_ids = self.unique_segs(segm, ignore_ids=[0])
            assert num_segs <= self.max_objects, f"Number of segments {num_segs} is greater than max_objects {self.max_objects}"
            segmented_objects = self.get_segm_objs(segm, unique_segm_ids)
            # these are used to generate the labels

            rgb = copy.deepcopy(obs['rgb'][view])
            if self.overlay_img:
                rgb = self.overlay_segs(rgb, segmented_objects)
            for seg in segmented_objects:
                rgb, seg = self.add_label(
                    rgb=rgb,
                    seg=seg,
                    position='center',
                    center_label_type=self.center_label_type,
                    sample_type=self.sample_type,
                    mode=self.sample_mode
                )

            obs['segm_objs'][view] = copy.deepcopy(segmented_objects)
            obs['o_rgb'][view] = rgb

        return obs

    def generate_data(
            self,
            skill: PromptBase,
            # text prompt
            task_prompt: str,
            meta_info: dict,
            # observation
            views: list[str],
            obs: dict,
            # specifies number of examples in the prompt
            n_exemplars: int = 0,
        ):
        text_list = []
        obs_list = []
        label_list = []
        assert n_exemplars == 0, f"n_exemplars {n_exemplars} not equal to 0"
        obs = self.generate_obs(
            obs=obs,
            views=views,
            ooi=meta_info['object_of_interest']
        )
        text = skill.generate_text_prompt(
            task_prompt=task_prompt,
            meta_info=meta_info,
            obs=obs,
        )
        label = skill.generate_labels(
            obs=obs,
            meta_info=meta_info
        )
        obs_list.append(obs)
        text_list.append(text)
        label_list.append(label)

        assert len(obs_list) == len(text_list) == len(label_list), f"Length of obs_list {len(obs_list)}, text_list {len(text_list)}, label_list {len(label_list)} not equal"
        return obs_list, text_list, label_list


class DirectionOverlayNum(SegmentOverlayNum):

    def add_arrows_on_obs(
            self,
            seg_pt_key: str, # key to selected point in the segment
            obs: dict,
            views: list[str],
        ):

        obs['o_arrow'] = {}
        for view in views:
            obs['o_arrow'][view] = None
            rgb = copy.deepcopy(obs['rgb'][view])
            segm_objs = obs['segm_objs'][view]

            found=False
            for seg in segm_objs:
                print(seg.sample_pts.keys())
                if seg_pt_key not in seg.sample_pts.keys():
                    continue
                found=True
                break
            assert found, f"Segment point key {seg_pt_key} not found in any of the segments"
            print(colored(f"Adding arrow to the segment {seg.segm_id} with label {seg_pt_key}", 'blue'))
            # take the point from the segment
            pt = seg.sample_pts[seg_pt_key]
            # add the arrows on the image in 4 directions
            rgb = U.add_arrows_on_image(
                    rgb,
                    pt,
                    directions=['clockwise'],
                    length=400,
                    thickness=5,
                    bgr_color=(0, 0, 255),
            )
            obs['o_arrow'][view] = rgb
        return obs

    def generate_obs(
            self,
            obs: dict,
            seg_pt_key: str,
            views: list[str],
            ooi: list[str] = ["all objects"],
            resample_pts: bool = False,
        ):
        '''
            Given an observation in dict,
            - Take the segm image and count the number of unique segments.
            - For each unique segment,
                create a mask
                unique_segment_mask
                center of segment
                area of segment
            - overlay it on the rgb image.
            - Add the overlayed rgb image to the observation with key o_rgb.
        '''
        if resample_pts:
            obs = super().generate_obs(obs, views, ooi)

        obs['dir_rgb'] = {}
        for view in views:
            segm = obs['segm'][view]
            obs['dir_rgb'][view] = None
            segmented_objects = obs['segm_objs'][view]
            rgb = copy.deepcopy(obs['rgb'][view])
            if self.overlay_img:
                rgb = self.overlay_segs(rgb, segmented_objects)
            found = False
            for seg in segmented_objects:
                if seg_pt_key in seg.sample_pts.keys():
                    rgb, seg = self.plot_points_on_image(
                        rgb=rgb,
                        seg=seg,
                        points=[seg.sample_pts[seg_pt_key]],
                        center_label_type='char',
                    )
                    found = True
                    break
            assert found, f"Segment point key {seg_pt_key} not found in any of the segments. Available keys are {[seg.sample_pts.keys() for seg in segmented_objects]}"
            obs['dir_rgb'][view] = rgb
        return obs

    def generate_data(
            self,
            skill: PromptBase,
            # text prompt
            task_prompt: str,
            meta_info: dict,
            # observation
            views: list[str],
            obs: dict,
            # specifies number of examples in the prompt
            n_exemplars: int = 0,
        ):
        text_list = []
        obs_list = []
        label_list = []
        assert n_exemplars == 0, f"n_exemplars {n_exemplars} not equal to 0"
        obs = self.generate_obs(
            obs=obs,
            views=views,
            ooi=meta_info['object_of_interest'],
            seg_pt_key=meta_info['point_of_interaction'],
        )
        text = skill.generate_text_prompt(
            task_prompt=task_prompt,
            meta_info=meta_info,
            obs=obs,
        )
        label = skill.generate_labels(
            obs=obs,
            meta_info=meta_info
        )
        obs_list.append(obs)
        text_list.append(text)
        label_list.append(label)

        assert len(obs_list) == len(text_list) == len(label_list), f"Length of obs_list {len(obs_list)}, text_list {len(text_list)}, label_list {len(label_list)} not equal"
        return obs_list, text_list, label_list
