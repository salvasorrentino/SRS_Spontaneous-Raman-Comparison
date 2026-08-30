import numpy as np
from skimage.transform import rotate
from skimage.feature import match_template

def find_best_angle_and_position(big_srs, small_cf,
                                 angle_range=(-3, 3),
                                 angle_step=0.25):
    Hc, Wc = small_cf.shape

    # 1) pad big SRS
    pad_y, pad_x = Hc, Wc
    big_padded = np.pad(big_srs,
                        ((pad_y, pad_y), (pad_x, pad_x)),
                        mode="constant", constant_values=0.0).T

    best_score = -np.inf
    best_angle = 0.0
    best_pos = (0, 0)

    angles = np.arange(angle_range[0], angle_range[1] + 1e-9, angle_step)

    for ang in angles:
        # rotate SRS around its center; keep same shape as padded image
        srs_rot = rotate(big_padded, ang,
                         resize=False,
                         preserve_range=True,
                         order=1)

        # template match: where does small_cf best match inside srs_rot?
        result = match_template(srs_rot, small_cf)
        ij = np.unravel_index(np.argmax(result), result.shape)
        score = result[ij]

        if score > best_score:
            best_score = score
            best_angle = float(ang)
            best_pos = ij  # (y0, x0) in padded rotated coordinates

    return best_angle, best_pos, (pad_y, pad_x)