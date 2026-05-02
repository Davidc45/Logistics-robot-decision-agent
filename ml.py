"""
ML / Data Module – Student 2 (Data / ML Engineer)

Loads robot_delay_data.csv, trains a Decision Tree classifier, and
predicts the delay_level for each candidate route produced by the
search module.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Tuple


class DelayPredictor:
    """Train once, predict many times."""

    FEATURE_COLS = ["time_of_day", "zone_type", "congestion_level", "distance"]
    TARGET_COL = "delay_level"

    def __init__(self, csv_path: str = "data/robot_delay_data.csv"):
        self.data = pd.read_csv(csv_path)
        self.encoders: Dict[str, LabelEncoder] = {}
        self.model = DecisionTreeClassifier(random_state=42)
        self._train()

    def _train(self):
        df = self.data.copy()
        for col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        X = df[self.FEATURE_COLS]
        y = df[self.TARGET_COL]
        self.model.fit(X.values, y)

    def predict(
        self,
        time_of_day: str,
        zone_type: str,
        congestion_level: str,
        distance: str,
    ) -> str:
        """Return predicted delay_level as a human-readable string."""
        sample = [
            self.encoders["time_of_day"].transform([time_of_day])[0],
            self.encoders["zone_type"].transform([zone_type])[0],
            self.encoders["congestion_level"].transform([congestion_level])[0],
            self.encoders["distance"].transform([distance])[0],
        ]
        pred = self.model.predict([sample])[0]
        return self.encoders[self.TARGET_COL].inverse_transform([pred])[0]

    def predict_for_route(self, route_info: dict, time_of_day: str) -> str:
        """
        Given a route analysis dict (from search.generate_candidate_routes)
        and the current time_of_day, determine the dominant zone/congestion
        and predict delay.
        """
        zones = route_info["zones_visited"]
        dominant_zone = max(zones, key=zones.get)

        cong = route_info["congestion_summary"]
        dominant_congestion = max(cong, key=cong.get)

        distance = route_info["distance_category"]

        return self.predict(time_of_day, dominant_zone, dominant_congestion, distance)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    predictor = DelayPredictor()
    print("Model trained on", len(predictor.data), "samples")
    print()

    test_cases = [
        ("morning", "normal", "low", "short"),
        ("evening", "high_traffic", "high", "long"),
        ("night", "restricted", "medium", "medium"),
    ]
    for tod, zt, cl, dist in test_cases:
        result = predictor.predict(tod, zt, cl, dist)
        print(f"  {tod}, {zt}, {cl}, {dist}  ->  delay = {result}")
