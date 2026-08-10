import cv2
import numpy as np

# Degrees of freedom each model is allowed to fit. Fewer is steadier: a model
# with more freedom than the footage needs spends the surplus fitting noise,
# which shows up as per-frame wobble in the composite.
ALIGN_MODELS = ["similarity", "affine", "homography"]


def to_u8(t):
    return (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def compute_transform(gray1, gray2, mask1=None, mask2=None, max_features=2000,
                      match_count=0, model="similarity", ratio=0.75):
    """Transform mapping gray1 coordinates -> gray2 coordinates, as a 3x3 matrix.

    mask1/mask2 restrict SIFT feature detection to white (255) regions of each
    image, e.g. to exclude areas that don't correspond between frames.

    Correspondences are filtered by Lowe's ratio test, which keeps a match only
    when it is decisively better than the runner-up. Selecting on how ambiguous
    a match is beats keeping a fixed count of the smallest descriptor distances:
    the latter has no idea whether the matches it kept were reliable, so raising
    the count just feeds the estimator a longer tail of bad ones.

    match_count caps the survivors (0 = keep all, which is what you want).
    """
    detector = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = detector.detectAndCompute(gray1, mask1)
    kp2, des2 = detector.detectAndCompute(gray2, mask2)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    good = []
    for pair in cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])

    if match_count and len(good) > match_count:
        good = sorted(good, key=lambda m: m.distance)[:match_count]
    if len(good) < 4:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if model == "homography":
        # MAGSAC copes with the wide inlier spread 8 free parameters produce.
        M, _ = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)
        return M

    estimator = cv2.estimateAffinePartial2D if model == "similarity" else cv2.estimateAffine2D
    A, _ = estimator(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return None if A is None else np.vstack([A, [0.0, 0.0, 1.0]])


# Kept so the old name still resolves; the default model matches the old behaviour.
def compute_homography(gray1, gray2, mask1=None, mask2=None, max_features=2000,
                       match_count=50):
    return compute_transform(gray1, gray2, mask1, mask2, max_features,
                             match_count, model="homography")


def _corners(size):
    w, h = size
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def _refit(size, pts):
    """Rebuild a 3x3 from where it sends the four frame corners (exact for 4 pairs)."""
    M, _ = cv2.findHomography(_corners(size), pts.reshape(-1, 1, 2), 0)
    return M


def _trajectories(transforms, size):
    """Where each transform sends the frame corners, for the frames that solved."""
    idx = [i for i, M in enumerate(transforms) if M is not None]
    if not idx:
        return [], None
    pts = np.stack([
        cv2.perspectiveTransform(_corners(size), transforms[i]).reshape(4, 2) for i in idx
    ])
    return idx, pts


def smooth_transforms(transforms, window, size):
    """Temporally smooth a sequence of transforms with a centred moving average.

    Averaging matrix entries directly is meaningless — the entries aren't
    independent — so each transform is reduced to the four corner positions it
    produces, those trajectories are smoothed, and the transform is refitted.
    """
    if window < 2:
        return transforms
    idx, pts = _trajectories(transforms, size)
    if pts is None or len(idx) < 2:
        return transforms

    k = window // 2
    out = list(transforms)
    for j, i in enumerate(idx):
        out[i] = _refit(size, pts[max(0, j - k): j + k + 1].mean(axis=0))
    return out


def lock_transforms(transforms, size):
    """Collapse to one transform for the whole clip — the per-frame median.

    Correct only when neither side moves, in which case every frame's estimate
    is measuring the same thing and the spread between them is pure noise.
    """
    idx, pts = _trajectories(transforms, size)
    if pts is None:
        return transforms
    locked = _refit(size, np.median(pts, axis=0))
    return [locked] * len(transforms)
