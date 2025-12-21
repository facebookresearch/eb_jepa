from dataclasses import fields

import numpy as np
import torch
import yaml
from scipy.stats import truncnorm


def update_config_from_yaml(config_class, yaml_data):
    """
    Create an instance of `config_class` using default values, but override
    fields with those provided in `yaml_data`.
    """
    config_field_names = {f.name for f in fields(config_class)}
    relevant_yaml_data = {
        key: value for key, value in yaml_data.items() if key in config_field_names
    }
    return config_class(**relevant_yaml_data)


def generate_wall_layouts(wall_config):
    """
    Generate possible layouts for train and validation
    """
    img_size = wall_config.img_size
    wall_padding = wall_config.wall_padding
    door_padding = wall_config.door_padding

    fix_wall = wall_config.fix_wall
    fix_door_location = wall_config.fix_door_location
    fix_wall_location = wall_config.fix_wall_location

    def extract_min_max(code):
        vals = [int(x) for x in code.split("-") if x]
        if len(vals) == 2:
            min_val, max_val = vals[0], vals[1]
        elif len(vals) == 1:
            min_val, max_val = vals[0], vals[0]
        else:
            return []
        return list(range(min_val, max_val + 1))

    exclude_wall_train = extract_min_max(wall_config.exclude_wall_train)
    exclude_door_train = extract_min_max(wall_config.exclude_door_train)
    only_wall_val = extract_min_max(wall_config.only_wall_val)
    only_door_val = extract_min_max(wall_config.only_door_val)

    "Arrays must all be empty or all be non-empty"
    assert all(
        len(arr) == 0
        for arr in [
            exclude_wall_train,
            exclude_door_train,
            only_wall_val,
            only_door_val,
        ]
    ) or all(
        len(arr) > 0
        for arr in [
            exclude_wall_train,
            exclude_door_train,
            only_wall_val,
            only_door_val,
        ]
    )

    if fix_wall:
        layouts = {
            f"v_wall{fix_wall_location}_door{fix_door_location}": {
                "type": "v",
                "wall_pos": fix_wall_location,
                "door_pos": fix_door_location,
            }
        }
        return layouts, None

    wall_x_values = list(range(wall_padding, img_size - wall_padding))
    door_y_values = list(range(door_padding, img_size - door_padding))

    layouts = {}
    other_layouts = {}

    for wall_pos in wall_x_values:
        for door_pos in door_y_values:
            code = f"wall{wall_pos}_door{door_pos}"

            to_exclude_for_train = (
                wall_pos in exclude_wall_train and door_pos in exclude_door_train
            )
            to_include_for_val = wall_pos in only_wall_val and door_pos in only_door_val

            if not to_exclude_for_train:
                layouts[f"v_{code}"] = {
                    "type": "v",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }

                layouts[f"h_{code}"] = {
                    "type": "h",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }

            if to_include_for_val:
                other_layouts[f"v_{code}"] = {
                    "type": "v",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }

                other_layouts[f"h_{code}"] = {
                    "type": "h",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }

    if not wall_config.train and exclude_wall_train:
        return other_layouts, None
    else:
        return layouts, None


def sample_uniformly_between(a, b):
    """
    Generate tensor c of the same shape as a and b, where each element of c
    is sampled uniformly between a[i] and b[i].

    Args:
    - a (torch.Tensor): Input tensor of shape (bs).
    - b (torch.Tensor): Input tensor of shape (bs), where each element is guaranteed to be bigger than a[i].

    Returns:
    - c (torch.Tensor): Output tensor of the same shape as a and b, with elements sampled uniformly.
    """
    # Generate random numbers from uniform distribution between a[i] and b[i]
    random_values = torch.FloatTensor(a.size()).uniform_(0, 1).to(a.device)

    # Scale the random values to fit the range [a[i], b[i]]
    c = a + (b - a) * random_values

    return c


def sample_truncated_norm(upper_bound, lower_bound, mean, std=1.4):
    """
    Generate tensor of the same shape as mean, where each element is
    sampled from a truncated norm distribution defined by upper_bound[i],
    lower_bound[i], and mean mean[i]
    """
    # Convert PyTorch tensors to NumPy
    upper_bound_np = upper_bound.cpu().float().numpy()
    lower_bound_np = lower_bound.cpu().float().numpy()
    mean_np = mean.cpu().float().numpy()

    # Initialize the output array
    samples = np.zeros_like(mean_np)

    # Sample from the truncated normal distribution
    for i in range(len(mean_np)):
        # Calculate the bounds in terms of standard deviations
        a, b = (
            (lower_bound_np[i] - mean_np[i]) / std,
            (upper_bound_np[i] - mean_np[i]) / std,
        )
        # Sample from truncnorm
        samples[i] = truncnorm.rvs(a, b, loc=mean_np[i], scale=std)

    # Convert back to PyTorch tensor
    return torch.from_numpy(samples)


def check_vertical_wall_intersect(pos1, pos2, wall_x, hole_y, door_space):
    check_intersection = (
        torch.sign(pos1[0] - wall_x) * torch.sign(pos2[0] - wall_x)
    ) <= 0.1
    if check_intersection:
        # print("found intersection at", i, j.item())
        d = pos2 - pos1
        # a and b are the line parameters fit to the last step
        a = d[1] / d[0]
        b = pos1[1] - a * pos1[0]
        # y is the intersection point of the wall plane
        y = a * wall_x + b
        # If the intersection point is in the hole, we are good
        # otherwise, we need to move the point back
        if (
            hole_y is None or y < hole_y - door_space or y > hole_y + door_space
        ):  # we're not in the hole
            return torch.tensor([wall_x, y]).to(pos1.device)  # we intersect
        else:
            return None  # we are in the hole and we overlap
    return None


def check_horizontal_wall_intersect(pos1, pos2, wall_y, hole_x, door_space):
    check_intersection = (
        torch.sign(pos1[1] - wall_y) * torch.sign(pos2[1] - wall_y)
    ) <= 0.1
    if check_intersection:
        d = pos2 - pos1
        a = d[1] / d[0]
        b = pos1[1] - a * pos1[0]
        x = (wall_y - b) / a
        if (
            hole_x is None or x < hole_x - door_space or x > hole_x + door_space
        ):  # we're not in the hole
            return torch.tensor([x, wall_y]).to(pos1.device)  # we intersect
        else:
            return None  # we are in the hole and we overlap
    return None


def check_wall_intersect(
    pos1,
    pos2,
    wall_x,
    hole_y,
    wall_width,
    door_space,
    border_wall_loc,
    img_size,
    add_noise=True,
):
    """
    Parameters:
        pos1: [2]
        pos2: [2]
        wall_x: []
        hole_y: []
        wall_width: int
        door_space: int
        border_wall_loc: int
        img_size: int
    Returns:
        intersect: [2]
        intersect_w_noise: [2]
    """

    # first, we check to see if the point bumps into the middle wall's width
    left_wall_corner, right_wall_corner = (
        wall_x - wall_width // 2,
        wall_x + wall_width // 2,
    )
    door_bot, door_top = hole_y - door_space, hole_y + door_space

    # check if it's moving upwards and crosses the door top horizontal line
    if pos2[1] - pos1[1] > 0 and pos2[1] > door_top and pos1[1] < door_top:
        # get x intercept with door top horizontal line
        intersect = check_horizontal_wall_intersect(
            pos1, pos2, door_top, None, door_space
        )
        # if x intercept occurs between the left and right wall
        if (
            intersect is not None
            and left_wall_corner <= intersect[0]
            and intersect[0] <= right_wall_corner
        ):
            # add downward noise and return early
            noise = torch.randn(2, device=pos1.device) * 0.5
            noise[1] = noise[1].abs() * -1
            return intersect, intersect + noise

    # check if it's moving downwards and croseses the door bot horizontal line
    if pos2[1] - pos1[1] < 0 and pos2[1] < door_bot and pos1[1] > door_bot:
        # get x intercept with door bot horizontal line
        intersect = check_horizontal_wall_intersect(
            pos1, pos2, door_bot, None, door_space
        )
        # if x intercept occurs between the left and right wall
        if (
            intersect is not None
            and left_wall_corner <= intersect[0]
            and intersect[0] <= right_wall_corner
        ):
            # add upward noise and return early
            noise = torch.randn(2, device=pos1.device) * 0.5
            noise[1] = noise[1].abs()
            return intersect, intersect + noise

    # next, we check to see if point bumps into border and wall proper
    left_wall, left_hole = border_wall_loc - 1, None
    right_wall, right_hole = img_size - border_wall_loc, None
    if wall_x > pos1[0]:
        right_wall, right_hole = wall_x - wall_width // 2, hole_y
    else:
        left_wall, left_hole = wall_x + wall_width // 2, hole_y

    top_wall, top_hole = border_wall_loc - 1, None
    bot_wall, bot_hole = img_size - border_wall_loc, None

    vertical_intersect = check_vertical_wall_intersect(
        pos1, pos2, left_wall, left_hole, door_space
    )
    if vertical_intersect is None:
        vertical_intersect = check_vertical_wall_intersect(
            pos1, pos2, right_wall, right_hole, door_space
        )

    horizontal_intersect = check_horizontal_wall_intersect(
        pos1, pos2, top_wall, top_hole, door_space
    )
    if horizontal_intersect is None:
        horizontal_intersect = check_horizontal_wall_intersect(
            pos1, pos2, bot_wall, bot_hole, door_space
        )

    if vertical_intersect is not None:
        sign = torch.sign(pos1[0] - vertical_intersect[0])
        vertical_noise = torch.randn(2, device=pos1.device) * 0.5
        vertical_noise[0] = vertical_noise[0].abs() * sign

    if horizontal_intersect is not None:
        sign = torch.sign(pos1[1] - horizontal_intersect[1])
        horizontal_noise = torch.randn(2, device=pos1.device) * 0.5
        horizontal_noise[1] = horizontal_noise[1].abs() * sign

    if vertical_intersect is not None and horizontal_intersect is not None:
        # return the intersection that happens first
        if torch.norm(pos1 - vertical_intersect) < torch.norm(
            pos1 - horizontal_intersect
        ):
            intersect = vertical_intersect
            noise = vertical_noise
        else:
            intersect = horizontal_intersect
            noise = horizontal_noise
    elif vertical_intersect is not None:
        intersect = vertical_intersect
        noise = vertical_noise
    elif horizontal_intersect is not None:
        intersect = horizontal_intersect
        noise = horizontal_noise
    else:
        return None, None

    intersect_w_noise = intersect + noise
    # we make sure after adding noise, we don't cross another wall
    intersect_w_noise[0] = torch.clamp(
        intersect_w_noise[0], min=left_wall, max=right_wall
    )
    intersect_w_noise[1] = torch.clamp(intersect_w_noise[1], min=top_wall, max=bot_wall)

    if intersect_w_noise[0] <= left_wall:
        intersect_w_noise[0] = left_wall + 0.3
    if intersect_w_noise[0] >= right_wall:
        intersect_w_noise[0] = right_wall - 0.3
    if intersect_w_noise[1] <= top_wall:
        intersect_w_noise[1] = top_wall + 0.3
    if intersect_w_noise[1] >= bot_wall:
        intersect_w_noise[1] = bot_wall - 0.3

    return intersect, intersect_w_noise
