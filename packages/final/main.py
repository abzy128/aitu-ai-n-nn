from __future__ import annotations

from final_anomaly.train import train_all


def main() -> None:
    metrics = train_all()
    print("Exported models:")
    for name in metrics["ranking_by_test_f1"]:
        item = metrics["models"][name]
        test = item["test"]
        print(
            f"- {name}: F1={test['f1']:.4f}, precision={test['precision']:.4f}, "
            f"recall={test['recall']:.4f}, artifact={item['artifact']}"
        )


if __name__ == "__main__":
    main()
