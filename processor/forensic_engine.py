import cv2
import numpy as np
import tensorflow as tf
from processor.image_utils import encode_to_base64


class ForensicEngine:

    @staticmethod
    def get_score_cam(img_tensor, model, layer_name="backbone_norm"):
        """
        Standalone Score-CAM generator.
        Used by Service for TTA and internally for cleaning masks.
        """
        try:
            real_model = model.encoder if hasattr(model, 'encoder') else model
            target_layer = real_model.get_layer(layer_name)

            cam_model = tf.keras.Model(
                real_model.input,
                [target_layer.output, real_model.output[0]]
            )
            conv_outputs, _ = cam_model(img_tensor, training=False)

            # Global Average Pooling over channels to get weights
            weights = tf.reduce_mean(conv_outputs[0], axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(
                weights, conv_outputs[0]), axis=-1).numpy()
            return np.maximum(cam, 0)
        except Exception as e:
            print(f"⚠️ Score-CAM Error: {e}")
            return None

    @staticmethod
    def _get_clean_mask(img_gray, model, img_tensor):
        """
        Refines scattered neural attention into a specific search zone.
        """
        try:
            # 1. Generate Raw Score-CAM
            raw_cam = ForensicEngine.get_score_cam(img_tensor, model)

            # 2. Normalize and Resize to Image Dimensions
            cam = cv2.normalize(raw_cam, None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)
            cam = cv2.resize(cam, (img_gray.shape[1], img_gray.shape[0]))

            # 3. Apply Strong Gaussian Blur (Fixes blocky artifacts)
            # This creates a 'soft' focus area that bridges the neural grid gaps.
            cam = cv2.GaussianBlur(cam, (21, 21), 0)

            # 4. Strict Binary Threshold
            # Higher values (100-120) create a tighter, more specific mask.
            _, mask = cv2.threshold(cam, 110, 255, cv2.THRESH_BINARY)

            return mask, cam
        except Exception as e:
            return None, None

    @staticmethod
    def generate_evidence(model, query_rgb, query_attention, match_path, match_tensor):
        """
        Screening Phase: RANSAC geometric proof within Neural Zones.
        """
        match_bgr = cv2.imread(match_path)
        if match_bgr is None:
            return None, 0, None

        gray_q = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)
        gray_m = cv2.cvtColor(cv2.cvtColor(
            match_bgr, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)

        # 1. Zone Setup: Reuse query attention, generate new mask for database image
        mask_q = cv2.resize(
            query_attention, (gray_q.shape[1], gray_q.shape[0])).astype(np.uint8)
        _, mask_q = cv2.threshold(mask_q, 100, 255, cv2.THRESH_BINARY)

        mask_m, raw_cam = ForensicEngine._get_clean_mask(
            gray_m, model, match_tensor)

        # 2. ORB tuned for minimalist logos (edgeThreshold=10 for 'SM' lines)
        orb = cv2.ORB_create(nfeatures=2000, edgeThreshold=10,
                             patchSize=31, fastThreshold=10)
        kp1, des1 = orb.detectAndCompute(gray_q, mask_q)
        kp2, des2 = orb.detectAndCompute(gray_m, mask_m)

        if des1 is None or des2 is None or len(kp1) < 4:
            cam_jet = cv2.applyColorMap(
                raw_cam, cv2.COLORMAP_JET) if raw_cam is not None else None
            return None, 0, encode_to_base64(cam_jet) if cam_jet is not None else None

        # 3. Geometric Verification (RANSAC)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]

        inliers = []
        if len(matches) >= 4:
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            _, ransac_mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, 5.0)
            if ransac_mask is not None:
                inliers = [matches[i] for i in range(
                    len(matches)) if ransac_mask.ravel().tolist()[i] == 1]

        # 4. Final Scoring & Calibration
        pts = len(inliers)
        # Threshold prevents 'lucky' low-point matches
        boost = (pts * 2.5) if pts >= 5 else 0

        # 5. Visualization Generation
        viz = cv2.drawMatches(query_rgb, kp1, cv2.cvtColor(
            match_bgr, cv2.COLOR_BGR2RGB), kp2, inliers, None, flags=2, matchColor=(0, 255, 0))
        cv2.putText(viz, f"Forensic Match: {
                    pts} pts", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        h_viz = cv2.applyColorMap(
            raw_cam, cv2.COLORMAP_JET) if raw_cam is not None else None

        return encode_to_base64(viz), round(boost, 2), encode_to_base64(h_viz) if h_viz is not None else None
