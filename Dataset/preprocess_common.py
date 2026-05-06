"""Joint StandardScaler on all circuits; append standardized log_degree; save processed features."""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(_ROOT, "Dataset")

COL_FANIN = 8
COL_FANOUT = 9
COL_DEPTH = 10
COL_DIST_IN = 11
COL_DIST_OUT = 12
COLS_TO_SCALE = [COL_FANIN, COL_FANOUT, COL_DEPTH, COL_DIST_IN, COL_DIST_OUT]


def main() -> None:
    cores = [d for d in os.listdir(_DATA_ROOT) if os.path.isdir(os.path.join(_DATA_ROOT, d))]
    print(f"Found {len(cores)} cores: {cores}")

    all_features = []
    core_data = {}

    for core in cores:
        core_path = os.path.join(_DATA_ROOT, core)
        f_path = os.path.join(core_path, "features.npy")
        if not os.path.exists(f_path):
            # Try specific naming if features.npy doesn't exist
            f_path = os.path.join(core_path, f"features_{core.lower()}.npy")
        
        if os.path.exists(f_path):
            feat = np.load(f_path, allow_pickle=False)
            all_features.append(feat)
            core_data[core] = feat
        else:
            print(f"Warning: No features found for {core}")

    if not all_features:
        print("No features to process.")
        return

    X_all = np.vstack(all_features)

    scaler = StandardScaler()
    scaler.fit(X_all[:, COLS_TO_SCALE])

    # Compute global log_degree stats
    fanin_all = X_all[:, COL_FANIN]
    fanout_all = X_all[:, COL_FANOUT]
    log_degree_all = np.log(1.0 + fanin_all + fanout_all)
    mean_ld = log_degree_all.mean()
    std_ld = log_degree_all.std()
    if std_ld < 1e-12:
        std_ld = 1.0

    CLIP_SIGMA = 3.0

    def transform_features(X: np.ndarray) -> np.ndarray:
        X_out = X.astype(np.float64, copy=True)
        X_out[:, COLS_TO_SCALE] = np.clip(
            scaler.transform(X[:, COLS_TO_SCALE]), -CLIP_SIGMA, CLIP_SIGMA
        )
        
        # Add standardized log degree
        fanin = X[:, COL_FANIN]
        fanout = X[:, COL_FANOUT]
        ld = np.log(1.0 + fanin + fanout)
        ld_std = np.clip((ld - mean_ld) / std_ld, -3.0, 3.0)
        
        return np.column_stack([X_out, ld_std])

    for core, feat in core_data.items():
        processed = transform_features(feat)
        out_path = os.path.join(_DATA_ROOT, core, f"features_processed.npy")
        np.save(out_path, processed)
        print(f"Processed {core}: {processed.shape}")

    print("\nGlobal log_degree mean:", float(mean_ld))
    print("Global log_degree std:", float(std_ld))


if __name__ == "__main__":
    main()
