import numpy as np

def pad_image(
        image: np.ndarray,
        target_size: tuple[int, int]
    ) -> np.ndarray:
    """
    Pad the image with zeros to make it the target size.
    The original image is centered in the padded image.
    """
    assert image.shape[0] <= target_size[0] and image.shape[1] <= target_size[1], f"Image size {image.shape} is larger than target size {target_size}"
    fill_value = 0
    if len(image.shape) == 3:
        target_size = (*target_size, image.shape[-1])
        fill_value = 255
    # padded_image is filled with 255 values
    padded_image = np.full(target_size, fill_value, dtype=image.dtype)
    start_x = (target_size[0] - image.shape[0]) // 2
    start_y = (target_size[1] - image.shape[1]) // 2
    padded_image[start_x:start_x+image.shape[0], start_y:start_y+image.shape[1]] = image
    return padded_image

def preprocess_vima_obs(
        obs: dict,
        valid_views: list[str],
        target_size: tuple[int, int] = (256, 256),
        use_gt_seg: bool = True,
        channel_last: bool = True,
    ) -> dict:
    """
        Process the observation from VIMA environment.
        The observation is a dictionary with the keys (superset of valid_views).

        Convert the image 128x256 to 256x256 by padding the image with zeros.
    """
    assert use_gt_seg, "not using use_gt_seg is not implemented"

    # Pad the images to make them the target size
    for m in obs.keys():
        if m == 'ee':
            continue
        for view in valid_views:
            if ('rgb' in m) and channel_last and (obs[m][view].shape[0] == 3):
                obs[m][view] = obs[m][view].transpose(1, 2, 0)
            obs[m][view] = pad_image(obs[m][view], target_size)

    return obs
