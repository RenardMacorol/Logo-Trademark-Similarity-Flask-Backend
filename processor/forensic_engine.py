import cv2
import numpy as np
from processor.image_utils import encode_to_base64


class ForensicEngine:
    @staticmethod
    def generate_evidence(query_rgb, match_path, ratio=0.75):
        """
        Stage 2: Spatial Verification (ORB + FLANN)
        """
        # 1. Load match image from path
        match_bgr = cv2.imread(match_path)
        if match_bgr is None:
            return None, 0
        match_rgb = cv2.cvtColor(match_bgr, cv2.COLOR_BGR2RGB)

        # 2. Pre-processing (Devane 2025 improvement)
        gray1 = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(match_rgb, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
        gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)

        # 3. ORB Setup
        orb = cv2.ORB_create(nfeatures=1500)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return None, 0

        # 4. FLANN Matcher (LSH Index)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # 5. Ratio Test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good_matches.append(m)

        # 6. Calculate Forensic Score (Percentage)
        # Ratio of good matches over query complexity
        forensic_score = (len(good_matches) / len(kp1)) * \
            100 if len(kp1) > 0 else 0
        # Adaptive scaling: More than 40 good matches is usually a confirmed clone
        confidence_boost = min(100.0, forensic_score * 5)

        # 7. Draw Visualization
        viz_img = cv2.drawMatches(query_rgb, kp1, match_rgb, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                  matchColor=(0, 255, 0))

        return encode_to_base64(viz_img), round(confidence_boost, 2)
