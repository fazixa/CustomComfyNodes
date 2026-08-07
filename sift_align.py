import cv2
import numpy as np


def to_u8(t):
    return (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def compute_homography(gray1, gray2, mask1=None, mask2=None, max_features=2000, match_count=50):
    """Homography mapping gray1 coordinates -> gray2 coordinates.

    mask1/mask2 restrict SIFT feature detection to white (255) regions of
    each image, e.g. to exclude areas that don't correspond between frames.
    """
    detector = cv2.SIFT_create(nfeatures=max_features)
    matcher  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    kp1, des1 = detector.detectAndCompute(gray1, mask1)
    kp2, des2 = detector.detectAndCompute(gray2, mask2)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)[:match_count]
    if len(matches) < 4:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H
