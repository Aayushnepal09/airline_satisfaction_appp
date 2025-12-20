# test_inference.py
"""Quick sanity check: load the saved model(s) and run inference.

Run:
    python test_inference.py
"""

from pathlib import Path

import joblib
import pandas as pd


CLASS_LABELS = {
    0: "neutral or dissatisfied",
    1: "satisfied",
}


def test_model(model_path: Path) -> bool:
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print('='*60)

    try:
        print("Loading model...", end=" ")
        model = joblib.load(model_path)
        print("✓")

        if hasattr(model, "named_steps"):
            print(f"Pipeline steps: {list(model.named_steps.keys())}")

        sample_data = pd.DataFrame(
            {
                "gender": ["Female", "Male"],
                "customer_type": ["Loyal Customer", "disloyal Customer"],
                "age": [35, 29],
                "type_of_travel": ["Business travel", "Personal Travel"],
                "travel_class": ["Business", "Eco"],
                "flight_distance": [1200, 450],
                "inflight_wifi_service": [4, 2],
                "departure_arrival_time_convenient": [3, 2],
                "ease_of_online_booking": [3, 2],
                "gate_location": [2, 3],
                "food_and_drink": [3, 2],
                "online_boarding": [4, 3],
                "seat_comfort": [4, 3],
                "inflight_entertainment": [4, 2],
                "on_board_service": [4, 3],
                "leg_room_service": [4, 3],
                "baggage_handling": [4, 3],
                "checkin_service": [4, 3],
                "inflight_service": [4, 3],
                "cleanliness": [4, 3],
                "departure_delay_minutes": [0, 15],
                "arrival_delay_minutes": [0, 20],
            }
        )

        print("Running inference...", end=" ")
        preds = model.predict(sample_data)
        print("✓")

        print("\nPredictions:")
        for i, p in enumerate(preds, 1):
            p_int = int(p)
            print(f"  Sample {i}: {p_int} ({CLASS_LABELS.get(p_int)})")

        print(f"\n✓ {model_path.name} - SUCCESS")
        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False


def main() -> bool:
    base_dir = Path(__file__).resolve().parent
    models = [
        base_dir / "models" / "global_best_model.pkl",
        base_dir / "models" / "global_best_model_optuna.pkl",
    ]

    print("Testing saved models...")
    results = [test_model(p) for p in models]

    print(f"\n{'='*60}")
    print(f"SUMMARY: {sum(results)}/{len(results)} models passed")
    print('='*60)

    return all(results)


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
