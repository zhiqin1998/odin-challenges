# adapted from https://github.com/AImageLab-zip/ToothFairy/tree/main/ToothFairy3/Interactive-Segmentation
import numpy as np


def sample_clicks_from_mask(mask, num_clicks=5, noise_level=5, x_noise=5):
    """Sample click points from a binary mask with less uniform X sampling and centered Y, Z."""
    indices = np.argwhere(mask)

    if indices.size == 0:
        return []  # No valid clicks found

    x_coords = indices[:, 0]  # Axial slices
    y_coords = indices[:, 1]  # Y-coordinates
    z_coords = indices[:, 2]  # Z-coordinates

    # Find min/max X, but start slightly later (0-5 slices offset)
    x_min, x_max = x_coords.min(), x_coords.max()
    x_start = np.random.randint(x_min, x_min + 6)
    x_end = np.random.randint(x_max - 5, x_max + 1)

    # Define valid X-values where the mask has valid points
    valid_x = [el for el in np.arange(x_min, x_max + 1) if np.any(mask[el])]

    # Sample X-values with approximate uniformity while ensuring validity
    sampled_x = np.linspace(x_start, x_end, num=num_clicks, dtype=int) if num_clicks > 1 else np.asarray([(x_start + x_end) / 2])
    sampled_x = np.array([valid_x[np.abs(np.array(valid_x) - x).argmin()] for x in sampled_x])

    # Add noise and ensure X values remain within valid_x
    sampled_x += np.random.randint(-x_noise, x_noise + 1, size=num_clicks)
    sampled_x = np.clip(sampled_x, x_start, x_end)
    sampled_x = np.array([valid_x[np.abs(np.array(valid_x) - x).argmin()] for x in sampled_x])

    # Ensure first and last clicks are exactly at the adjusted start and end points
    sampled_x[0], sampled_x[-1] = x_start, x_end
    if x_start not in valid_x:
        sampled_x[0] = x_min
    if x_end not in valid_x:
        sampled_x[-1] = x_max

    # for x in sampled_x:
    #     assert x in valid_x # check if sampled slices contain a label

    click_points = []
    for x in sampled_x:
        valid_points = indices[x_coords == x]  # Get all (Y, Z) for this X slice
        assert valid_points.size > 0
        # Compute center of Y and Z for this slice
        y_center = np.median(valid_points[:, 1]).astype(int)
        z_center = np.median(valid_points[:, 2]).astype(int)

        # # Add small noise while keeping in bounds
        # y = np.clip(y_center + np.random.randint(-noise_level, noise_level + 1), y_coords.min(), y_coords.max())
        # z = np.clip(z_center + np.random.randint(-noise_level, noise_level + 1), z_coords.min(), z_coords.max())

        # Add small noise while keeping in bounds
        for _ in range(3):  # Try n times to find a valid (y, z)
            y = np.clip(y_center + np.random.randint(-noise_level, noise_level + 1), y_coords.min(), y_coords.max())
            z = np.clip(z_center + np.random.randint(-noise_level, noise_level + 1), z_coords.min(), z_coords.max())
            if mask[x, y, z]:
                break
        else:
            # If no valid point is sampled with perturbation, sample one randomly
            y, z = valid_points[np.random.randint(valid_points.shape[0]), 1:]
            # warnings.warn('Using a uniform point since center perturbation always led to an invalid point...')
        click_points.append([int(el) for el in [x, y, z]])

        # assert mask[x, y, z] # make sure simulated click is in IAC
    return click_points
