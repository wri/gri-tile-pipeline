"""Temporal resampling to 24 biweekly composites.

Ported from reference ``downloading/utils.py`` ``calculate_and_save_best_images``.

Takes N images with their day-of-year dates and produces 24 evenly-spaced
(every 15 days) composites using weighted-nearest-neighbor interpolation.
This ensures the Whittaker smoother always gets 24 inputs regardless of
acquisition count or temporal distribution.
"""

from __future__ import annotations

import numpy as np


def resample_to_biweekly(
    img_bands: np.ndarray,
    image_dates: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Interpolate input data to 24 biweekly composites.

    For each of the 24 target dates (every 15 days: 0, 15, 30, ..., 345),
    selects up to 2 prior and 2 after images and blends them with
    inverse-distance weighting.

    Args:
        img_bands: ``(N, H, W, B)`` input array.
        image_dates: ``(N,)`` day-of-year for each input timestep.

    Returns:
        ``(24, H, W, B)`` resampled array and max temporal gap distance.
    """
    image_dates = np.array(image_dates, dtype=np.float64)
    image_dates[image_dates < -100] = image_dates[image_dates < -100] % 365

    biweekly_dates = list(range(0, 360, 15))  # 24 target dates

    satisfactory_dates = image_dates.copy()
    selected_images = {}

    for target_day in biweekly_dates:
        distances = np.array([(d - target_day) for d in satisfactory_dates])

        # Find up to 2 prior images (within 100 days of closest)
        prior = distances[np.where(distances < 5)][-2:]
        if prior.shape[0] > 0:
            prior = np.array(prior[prior > (-100 + np.max(prior))]).flatten()

        # Find up to 2 after images (within 100 days of closest)
        after = distances[np.where(distances >= -5)][:2]
        if after.shape[0] > 0:
            after = np.array(after[after < (100 + np.min(after))])

        # Handle missing prior or after with wraparound
        prior_flag = 0
        after_flag = 0
        if len(prior) == 0:
            if np.min(satisfactory_dates) >= 90:
                prior = distances[-1:]
                prior_flag = 365
            else:
                prior = after
        if len(after) == 0:
            if np.max(satisfactory_dates) <= 270:
                after = distances[:1]
                after_flag = 365
            else:
                after = prior

        # Compute inverse-distance weights
        prior_calc = abs(prior - prior_flag)
        after_calc = abs(after + after_flag)
        prior_calc = np.maximum(prior_calc, 1.0)
        after_calc = np.maximum(after_calc, 1.0)

        total_dist = np.sum(np.concatenate([prior_calc, after_calc]))
        if total_dist == 0:
            prior_calc += 1
            after_calc += 1
            total_dist = np.sum(np.concatenate([prior_calc, after_calc]))

        closest_dist = np.maximum(abs(prior_calc[-1]) + abs(after_calc[0]), 2)

        prior_mults = abs(1 - (abs(prior_calc) / closest_dist))
        after_mults = abs(1 - (abs(after_calc) / closest_dist))

        if len(prior_mults) == 2:
            prior_mults[0] = abs((prior_calc[1] / prior_calc[0]) * prior_mults[1])
        if len(after_mults) == 2:
            after_mults[1] = abs((after_calc[0] / after_calc[1]) * after_mults[0])

        # Normalize weights to sum to 1
        divisor = np.sum(np.concatenate([abs(prior_mults), abs(after_mults)]))
        if divisor > 0:
            prior_ratio = prior_mults / divisor
            after_ratio = after_mults / divisor
        else:
            prior_ratio = np.ones_like(prior_mults) / (len(prior_mults) + len(after_mults))
            after_ratio = np.ones_like(after_mults) / (len(prior_mults) + len(after_mults))

        # Find image indices
        prior_dates = target_day + prior
        prior_idx = [i for i, val in enumerate(image_dates) if val in prior_dates]
        prior_idx = sorted(list(set(np.array(prior_idx).reshape(-1))))
        after_dates = target_day + after
        after_idx = [i for i, val in enumerate(image_dates) if val in after_dates]
        after_idx = sorted(list(set(np.array(after_idx).reshape(-1))))

        if len(after_idx) > 2:
            after_idx = after_idx[-2:]
        if len(prior_idx) > 2:
            prior_idx = prior_idx[:2]

        selected_images[target_day] = {
            'image_date': np.concatenate([prior_dates, after_dates]).flatten(),
            'image_ratio': [prior_ratio, after_ratio],
            'image_idx': [prior_idx, after_idx],
        }

    # Compute max temporal gap
    max_distance = 0
    for key in sorted(selected_images.keys()):
        info = selected_images[key]
        if len(info['image_date']) == 2:
            dist = np.min(info['image_date'][1]) - np.max(info['image_date'][0])
            if dist > max_distance:
                max_distance = dist

    # Build 24 composites
    keep_steps = []
    for key in sorted(selected_images.keys()):
        info = selected_images[key]

        if len(info['image_idx']) == 1 or (len(info['image_idx'][0]) == 0 and len(info['image_idx'][1]) == 0):
            # Fallback: use median of all images
            step = np.median(img_bands, axis=0)
        elif len(info['image_idx']) >= 2:
            # Weighted combination of prior and after images
            step1 = img_bands[info['image_idx'][0]] if len(info['image_idx'][0]) > 0 else img_bands[[0]]
            if len(step1.shape) == 3:
                step1 = step1[np.newaxis]
            step1mult = np.array(info['image_ratio'][0], dtype=np.float32)
            step1mult = step1mult[..., np.newaxis, np.newaxis, np.newaxis]
            step1 = np.sum(step1.copy() * step1mult, axis=0)

            step2 = img_bands[info['image_idx'][1]] if len(info['image_idx'][1]) > 0 else img_bands[[-1]]
            if len(step2.shape) == 3:
                step2 = step2[np.newaxis]
            step2mult = np.array(info['image_ratio'][1], dtype=np.float32)
            step2mult = step2mult[..., np.newaxis, np.newaxis, np.newaxis]
            step2 = np.sum(step2.copy() * step2mult, axis=0)

            step = step1 + step2
        else:
            step = np.median(img_bands, axis=0)

        keep_steps.append(step)

    return np.stack(keep_steps), int(max_distance)
