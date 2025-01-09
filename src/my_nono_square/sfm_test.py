import cv2
import numpy as np
import matplotlib.pyplot as plt

def capture_two_images_from_camera():
    """
    Capture two images from the default camera (index=0).
    We'll ask the user to press space to capture each image.
    
    Returns:
      img1, img2: BGR images (numpy arrays)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open camera.")
        return None, None

    print("Press SPACE to capture the first image.")
    img1 = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Couldn't read from camera.")
            break
        cv2.imshow("Live Feed - Press SPACE for Image 1", frame)

        # Wait for key press
        key = cv2.waitKey(1)
        if key & 0xFF == ord(' '):
            # SPACE pressed, capture the frame
            img1 = frame.copy()
            print("Captured the first image.")
            break
        elif key & 0xFF == ord('q'):
            # Quit
            break

    if img1 is None:
        cap.release()
        cv2.destroyAllWindows()
        return None, None

    print("Press SPACE to capture the second image.")
    img2 = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Couldn't read from camera.")
            break
        cv2.imshow("Live Feed - Press SPACE for Image 2", frame)

        # Wait for key press
        key = cv2.waitKey(1)
        if key & 0xFF == ord(' '):
            img2 = frame.copy()
            print("Captured the second image.")
            break
        elif key & 0xFF == ord('q'):
            # Quit
            break

    cap.release()
    cv2.destroyAllWindows()
    return img1, img2


def detect_and_match_features(img1, img2):
    """
    Detect and match keypoints in two images using SIFT or ORB.
    We'll try SIFT first. If not available, switch to ORB.
    """
    # Prefer SIFT
    try:
        sift = cv2.SIFT_create()
    except:
        # If SIFT not available, use ORB
        print("[Info] SIFT not found, switching to ORB.")
        sift = cv2.ORB_create(nfeatures=1000)

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use FLANN or BFMatcher for matching
    # For SIFT, FLANN is typical
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # If descriptors are None (no features found), handle that
    if des1 is None or des2 is None:
        return None, None, [], [], []

    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2, kp1, kp2, good_matches


def estimate_motion_and_pose(pts1, pts2, K):
    """
    Estimate the essential matrix from matched keypoints and recover
    rotation and translation between two images, given the camera
    intrinsics K.
    """
    # If not enough points, return None
    if pts1 is None or pts2 is None or len(pts1) < 5:
        return None, None, None

    # Find essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None

    # Recover pose
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    return R, t, mask.reshape(-1)


def triangulate_points(pts1, pts2, R, t, K):
    """
    Triangulate 3D points given matched 2D points in two views.
    """
    I = np.eye(3, 3)
    zero = np.zeros((3, 1))
    P1 = K @ np.hstack((I, zero))  # [K|0]
    P2 = K @ np.hstack((R, t))     # [K|R|t]

    pts1_h = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    pts2_h = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]

    pts4D = cv2.triangulatePoints(P1, P2, pts1_h.T, pts2_h.T)
    pts3D = (pts4D[:3] / pts4D[3]).T

    return pts3D


def create_depth_map(img1, img2, K, R, t):
    """
    Create a rough depth map from two images by naive stereo matching.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # For demonstration, we do not rectify. We just run StereoSGBM as-is.
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,  # multiple of 16
        blockSize=5,
        uniquenessRatio=5,
        speckleWindowSize=5,
        speckleRange=2,
        disp12MaxDiff=1,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
    )

    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

    # Convert disparity to depth using a guessed baseline and the focal length
    focal = K[0, 0]
    baseline = 0.1  # 10 cm, just a guess
    with np.errstate(divide='ignore'):
        depth_map = (focal * baseline) / disparity
    depth_map[disparity <= 0] = 0

    return disparity, depth_map


def main():
    #--------------------------------------------------
    # 1. Capture two images from camera
    #--------------------------------------------------
    img1, img2 = capture_two_images_from_camera()
    if img1 is None or img2 is None:
        print("[Error] Failed to capture two valid images from camera.")
        return

    #--------------------------------------------------
    # 2. Dummy Intrinsic Matrix
    #--------------------------------------------------
    # For demonstration only. In real usage, calibrate the camera and use real intrinsics.
    focal_length = 800.0
    cx = 320.0
    cy = 240.0
    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]], dtype=np.float32)

    #--------------------------------------------------
    # 3. Detect & Match Features
    #--------------------------------------------------
    pts1, pts2, kp1, kp2, good_matches = detect_and_match_features(img1, img2)
    if pts1 is None or pts2 is None or len(pts1) < 8:
        print("[Error] Not enough good matches found.")
        return

    #--------------------------------------------------
    # 4. Estimate (R, t)
    #--------------------------------------------------
    R, t, inlier_mask = estimate_motion_and_pose(pts1, pts2, K)
    if R is None or t is None or inlier_mask is None:
        print("[Error] Could not estimate motion from the two images.")
        return

    # Filter inliers
    pts1_inliers = pts1[inlier_mask == 1]
    pts2_inliers = pts2[inlier_mask == 1]

    #--------------------------------------------------
    # 5. Triangulate
    #--------------------------------------------------
    points_3d = triangulate_points(pts1_inliers, pts2_inliers, R, t, K)
    print(f"Triangulated {points_3d.shape[0]} points (in 3D).")

    #--------------------------------------------------
    # 6. Create a Depth Map
    #--------------------------------------------------
    disparity, depth_map = create_depth_map(img1, img2, K, R, t)

    #--------------------------------------------------
    # 7. Visualize Results
    #--------------------------------------------------
    # 7.1. Show match inliers
    match_img = cv2.drawMatches(
        img1, [cv2.KeyPoint(pt[0], pt[1], 1) for pt in pts1_inliers],
        img2, [cv2.KeyPoint(pt[0], pt[1], 1) for pt in pts2_inliers],
        [cv2.DMatch(i, i, 0) for i in range(len(pts1_inliers))],
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Inlier Matches", match_img)

    # 7.2. Plot 3D points (requires matplotlib 3D)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', s=2)
    ax.set_title("Triangulated 3D Points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 7.3. Disparity / Depth
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(disparity, cmap='magma')
    plt.title("Disparity")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    depth_map_vis = np.clip(depth_map, 0, 1000)  # just clip for display
    plt.imshow(depth_map_vis, cmap='viridis')
    plt.title("Depth Map")
    plt.colorbar()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
