import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class LatentMapper:
    """
    Handles dimension reduction and coordinate scaling for Flutter UI rendering.
    """

    @staticmethod
    def map_to_2d(query_feat, neighbor_feats, background_feats):
        """
        Used for Search/Predict view.
        Maps the specific result context (Query vs Neighbors).
        """
        try:
            # 1. Standardize to 1D arrays
            q = np.array(query_feat).flatten()
            ns = [np.array(f).flatten() for f in neighbor_feats]
            bgs = [np.array(f).flatten() for f in background_feats]

            # 2. Combine and Stack
            all_feats = [q] + ns + bgs
            data_matrix = np.vstack(all_feats)

            # 3. PCA Projection to 2D
            pca = PCA(n_components=2)
            coords = pca.fit_transform(data_matrix)

            # 4. MOBILE OPTIMIZATION: Scale to 0.0 - 1.0 range
            scaler = MinMaxScaler(feature_range=(0, 1))
            coords_scaled = scaler.fit_transform(coords)

            # 5. Slice back into clean lists with rounding for JSON size efficiency
            return {
                "query": coords_scaled[0].round(4).tolist(),
                "neighbors": coords_scaled[1:len(ns)+1].round(4).tolist(),
                "background": coords_scaled[len(ns)+1:].round(4).tolist()
            }
        except Exception as e:
            print(f"⚠️ Latent Mapping Error: {e}")
            return {"query": [0.5, 0.5], "neighbors": [], "background": []}

    @staticmethod
    def map_to_2d_overview(embeddings, metadata):
        """
        Used for Discovery Galaxy view.
        Maps the entire dataset scope (PH, Global, or Both).
        """
        # 1. Validation: Siguraduhing may data at magkapares sila
        if embeddings is None or len(embeddings) < 2:
            print("⚠️ [MAPPER] Not enough embeddings to calculate PCA.")
            return []

        try:
            # Convert to numpy array once (important for memory efficiency)
            data_matrix = np.array(embeddings)

            # 2. PCA Reduction
            # Note: For 24k points, PCA is fast enough on a GTX 1650 / modern CPU.
            pca = PCA(n_components=2)
            coords = pca.fit_transform(data_matrix)

            # 3. Scaling for Flutter CustomPainter (0.0 to 1.0)
            scaler = MinMaxScaler(feature_range=(0, 1))
            coords_scaled = scaler.fit_transform(coords)

            results = []
            for i in range(len(coords_scaled)):
                # Cross-reference with metadata index
                m = metadata[i]

                results.append({
                    "x": float(coords_scaled[i][0].round(4)),
                    "y": float(coords_scaled[i][1].round(4)),
                    "image_path": m.get("image_path", ""),
                    # Contains brand_name, industry_domain, origin
                    "metadata": m.get("metadata", {})
                })

            print(f"✨ [MAPPER] Successfully mapped {
                  len(results)} points to 2D space.")
            return results

        except Exception as e:
            print(f"🔴 [MAPPER] Error during dimension reduction: {str(e)}")
            return []
