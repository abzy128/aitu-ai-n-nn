from __future__ import annotations

import pandas as pd

from final_anomaly.features import (
    LabelConfig,
    add_proxy_labels,
    build_feature_frame,
    chronological_split,
    make_sequences,
)


def test_proxy_labels_flag_low_power_and_large_jump() -> None:
    df = pd.DataFrame(
        {
            "DateTime": pd.date_range("2026-03-25", periods=4, freq="min"),
            "active_power": [24.0, 23.5, 6.0, 24.0],
        }
    )

    labeled = add_proxy_labels(df, LabelConfig(low_power_threshold=10.0, jump_threshold=5.0))

    assert labeled["proxy_anomaly"].tolist() == [0, 0, 1, 1]


def test_feature_frame_and_split_are_chronological() -> None:
    df = pd.DataFrame(
        {
            "DateTime": pd.date_range("2026-03-25", periods=100, freq="min"),
            "active_power": [20.0 + i * 0.01 for i in range(100)],
        }
    )
    features = build_feature_frame(add_proxy_labels(df))
    train_df, val_df, test_df = chronological_split(features)

    assert len(features) == 40
    assert train_df["DateTime"].max() < val_df["DateTime"].min()
    assert val_df["DateTime"].max() < test_df["DateTime"].min()


def test_make_sequences_aligns_label_to_window_endpoint() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0]).to_numpy()
    labels = pd.Series([0, 0, 1, 0]).to_numpy()

    windows, y = make_sequences(values, labels, length=3)

    assert windows.shape == (2, 3, 1)
    assert y.tolist() == [1, 0]
