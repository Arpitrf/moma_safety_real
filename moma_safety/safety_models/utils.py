import cv2
import torch
import open3d as o3d
import numpy as np
import groundingdino.datasets.transforms as T
import supervision as sv
from PIL import Image
from torchvision.ops import box_convert
import matplotlib
matplotlib.use("agg", force=True)
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from pointnet2.models import action_pointnet2_cls_ssg
from moma_safety.utils.env_variables import *
from moma_safety.utils.object_config import object_config as OC
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
# from pointnet2.data_utils.utils import generate_point_cloud_from_depth

BOX_THRESHOLD = 0.35 # 0.35 originally
TEXT_THRESHOLD = 0.25 # 0.25 originally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global clicked_points
        clicked_points.append([x, y])


def obtain_mask(env, object_name, select_object=False):
    obs = env._observation()
    depth = obs['tiago_head_depth']
    rgb = obs['tiago_head_image']
    if rgb.dtype != np.uint8:
        rgb = cv2.convertScaleAbs(rgb)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    img = rgb
    
    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # img_path = IMG_PATH
    # image_source, image = load_image(img_path)
    # breakpoint()
    sam2_predictor.set_image(img)
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()


    # TODO: Implement when not slecting object (i.e. using text query)
    if select_object:
        # choose a point
        global clicked_points
        cv2.namedWindow('color', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('color', mouseclick_callback)
        # color_im = cv2.imread(img_path)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        while True:
            if len(clicked_points) != 0:
                for point in clicked_points:
                    bgr_img = cv2.circle(bgr_img, point, 7, (0, 0, 255), 2)
            cv2.imshow('color', bgr_img)

            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyWindow('color')

        clicked_points = np.array(clicked_points)
        print("clicked_points: ", clicked_points)
        input_label = np.ones(len(clicked_points), dtype=int)

        masks, scores, logits = sam2_predictor.predict(
            point_coords=clicked_points,
            point_labels=input_label,
            box=None,
            multimask_output=False,
        )

        # breakpoint()
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        show_mask(masks, plt.gca())
        show_points(clicked_points, input_label, plt.gca())
        plt.axis('off')
        plt.savefig(f"resources/tmp_outputs/gsam_mask.jpg")
        # plt.show() 
        clicked_points = []
    # dependent on gorunded sam now (text-based)
    else:
        text = OC[object_name]["safety_model_text_description"]
        print("text: ", text)
        
        img_source = rgb.copy()
        img_PIL = Image.fromarray(img_source)

        transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        img_transformed, _ = transform(img_PIL, None)

        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=img_transformed,
            caption=text,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        
        # process the box prompt for SAM 2
        h, w, _ = img_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        print("boxes: ", input_boxes.shape)
        # to detect if object is found or not
        # if input_boxes.shape[0] > 0:

        # SAM part
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        sam2_predictor.set_image(img_source)
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)


        confidences = confidences.numpy().tolist()
        class_names = labels

        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        # Visualize
        class_names = labels
        class_ids = np.array(list(range(len(class_names))))
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img_source.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        plt.imsave(f"resources/tmp_outputs/dino_arm_expl_output.jpg", annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        plt.imsave(f"resources/tmp_outputs/gsam_arm_expl_output.jpg", annotated_frame)

    print("masks.shape:", masks.shape)
    return masks[0]