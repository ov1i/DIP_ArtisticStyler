import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def main():
    #-------------------------------------------------
    # 1. Open the camera
    #-------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open camera.")
        return

    #-------------------------------------------------
    # 2. Some parameters
    #-------------------------------------------------
    # Dummy intrinsics (focal length, principal point).
    # For real usage, calibrate your camera!
    focal_length = 700.0
    cx = 320.0
    cy = 240.0
    K = np.array([
        [focal_length, 0,           cx],
        [0,            focal_length, cy],
        [0,            0,            1]
    ], dtype=np.float32)

    # Create a SIFT or ORB detector
    try:
        detector = cv2.SIFT_create()
    except:
        print("[Info] SIFT not available, switching to ORB.")
        detector = cv2.ORB_create(nfeatures=2000)

    # FLANN (for SIFT) or BFMatcher (for ORB)
    use_flann = hasattr(detector, 'detectAndCompute') and 'SIFT' in str(type(detector))
    if use_flann:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # ORB - BFMatcher with Hamming norm
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # We store the last frame's keypoints, descriptors, etc.
    prev_kps = None
    prev_des = None
    prev_frame = None

    # We also store a "global pose" (R,t). We start at identity pose.
    R_global = np.eye(3, dtype=np.float64)
    t_global = np.zeros((3, 1), dtype=np.float64)

    # A small buffer to store some 3D points (for visualization or debugging)
    map_points_3d = deque(maxlen=20000)  # up to 20k points

    # We will store the previous projection matrix as well
    # For the first frame, we define P_prev = K[I|0]
    P_prev = K @ np.hstack((np.eye(3), np.zeros((3,1))))

    #-------------------------------------------------
    # 3. Main loop
    #-------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to read frame from camera.")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints, descriptors
        kps = detector.detect(gray, None)
        kps, des = detector.compute(gray, kps)

        # We'll skip the very first iteration until we have a previous frame
        if prev_frame is not None and prev_kps is not None and prev_des is not None:
            # 3.1. Match descriptors to previous frame
            if des is None or len(des) < 2:
                # No features or too few
                pass
            else:
                # KNN match if using FLANN or normal match if BF
                if use_flann:
                    matches_knn = matcher.knnMatch(prev_des, des, k=2)
                    good_matches = []
                    for m, n in matches_knn:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                else:
                    matches_knn = matcher.knnMatch(prev_des, des, k=2)
                    good_matches = []
                    for m, n in matches_knn:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                # Convert to point arrays
                pts_prev = np.float32([prev_kps[m.queryIdx].pt for m in good_matches])
                pts_curr = np.float32([kps[m.trainIdx].pt for m in good_matches])

                if len(pts_prev) >= 8:
                    # 3.2. Estimate essential matrix + recover pose
                    E, mask = cv2.findEssentialMat(
                        pts_prev, pts_curr, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
                    )
                    if E is not None:
                        _, R, t, mask_pose = cv2.recoverPose(E, pts_prev, pts_curr, K, mask=mask)
                        
                        # 3.3. Update global pose
                        # We interpret (R,t) as the motion from prev_frame to current_frame.
                        # So the new global pose = old pose * this motion
                        #  => R_global_new = R * R_global (actually we want R_global * R if we consider
                        #     the old coordinate system, but there's a convention detail).
                        # There's a sign or order detail to be careful with:
                        # Typically we do: [R_global|t_global] = [R|t]*[R_global|t_global].
                        # Let's do a naive approach:
                        R_global = R @ R_global
                        t_global = R @ t_global + t

                        # 3.4. Triangulate some points between last frame and current frame
                        # Projection matrix for previous frame
                        P1 = P_prev
                        # Projection for current frame:
                        #   R_curr = R_global, t_curr = t_global
                        P2 = K @ np.hstack((R_global, t_global))

                        # We only use inlier points for triangulation
                        inlier_idx = np.where(mask_pose.ravel() == 1)[0]
                        pts1_in = pts_prev[inlier_idx]
                        pts2_in = pts_curr[inlier_idx]

                        # We need shape (2, N) for cv2.triangulatePoints
                        pts1_2xN = pts1_in.T  # from (N,2) -> (2,N)
                        pts2_2xN = pts2_in.T

                        if pts1_2xN.shape[1] >= 8:
                            pts4D = cv2.triangulatePoints(P1, P2, pts1_2xN, pts2_2xN)
                            pts3D = (pts4D[:3] / pts4D[3]).T  # shape (N, 3)
                            
                            # Store some points in a deque
                            for p3d in pts3D:
                                map_points_3d.append(p3d)

                        # Update P_prev to the new camera projection
                        P_prev = P2
                        
                        # 3.5. (Optional) Visualize matches or do other debugging
                        # We'll draw just a small subset
                        match_vis = cv2.drawMatches(
                            prev_frame, prev_kps,
                            frame, kps,
                            good_matches[:50], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )
                        cv2.imshow("Matches (Current vs. Previous)", match_vis)

        # 3.6. Show the current frame with keypoints
        for kp in kps:
            cv2.circle(frame, (int(kp.pt[0]), int(kp.pt[1])), 2, (0,255,0), 1)
        cv2.imshow("Current Frame (with Keypoints)", frame)

        # 3.7. Press 'q' to quit
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        # 3.8. Prepare for next iteration
        prev_frame = frame.copy()
        prev_kps = kps
        prev_des = des

    cap.release()
    cv2.destroyAllWindows()

    #-------------------------------------------------
    # 4. Show final map (3D scatter)
    #-------------------------------------------------
    # Convert the deque of points to np array
    mp_np = np.array(map_points_3d)  # shape (N,3)
    if mp_np.shape[0] > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mp_np[:, 0], mp_np[:, 1], mp_np[:, 2], s=1, c='b')
        ax.set_title("Final 3D Map Points (sparse)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()


if __name__ == "__main__":
    main()
