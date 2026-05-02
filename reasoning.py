"""
Reasoning Module – Student 3 (Reasoning Engineer)

Logical decision rules, probability / uncertainty scoring, and
final route evaluation that selects the best route.
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Probability / Uncertainty
# ---------------------------------------------------------------------------

# Empirical delay probabilities conditioned on time_of_day.
# P(delay_level | time_of_day)  — derived from the dataset distribution.
DELAY_PROB = {
    "morning":   {"low": 0.50, "medium": 0.30, "high": 0.20},
    "afternoon": {"low": 0.40, "medium": 0.25, "high": 0.35},
    "evening":   {"low": 0.20, "medium": 0.30, "high": 0.50},
    "night":     {"low": 0.55, "medium": 0.30, "high": 0.15},
}

# Weights for the scoring formula
W_COST      = 0.30
W_DELAY     = 0.35
W_PROB      = 0.20
W_RESTRICT  = 0.15


def delay_probability(time_of_day: str, predicted_delay: str) -> float:
    """P(delay = predicted_delay | time_of_day)"""
    return DELAY_PROB.get(time_of_day, {}).get(predicted_delay, 0.33)


# ---------------------------------------------------------------------------
# Logical Decision Rules
# ---------------------------------------------------------------------------

def apply_rules(route_name: str, route_info: dict, predicted_delay: str) -> List[str]:
    """
    Evaluate 6 decision rules against a candidate route.
    Returns a list of rule verdicts (strings).
    """
    verdicts: List[str] = []

    # Rule 1: Restricted zone → reject
    if route_info["passes_restricted"]:
        verdicts.append("REJECT - route passes through a restricted zone")

    # Rule 2: High delay + long distance -> avoid
    if predicted_delay == "high" and route_info["distance_category"] == "long":
        verdicts.append("REJECT - predicted delay is high AND distance is long")

    # Rule 3: Medium congestion + short distance -> allow
    cong = route_info["congestion_summary"]
    if cong.get("medium", 0) > 0 and route_info["distance_category"] == "short":
        verdicts.append("ALLOW  - congestion is medium but route is short")

    # Rule 4: Normal zone + low delay -> prefer
    zones = route_info["zones_visited"]
    if zones.get("normal", 0) == sum(zones.values()) and predicted_delay == "low":
        verdicts.append("PREFER - all zones normal and delay is low")

    # Rule 5: High congestion on majority of path -> penalize
    total_cells = sum(cong.values())
    if total_cells > 0 and cong.get("high", 0) / total_cells > 0.5:
        verdicts.append("PENALIZE - more than half the path has high congestion")

    # Rule 6: Medium delay + medium distance -> acceptable
    if predicted_delay == "medium" and route_info["distance_category"] == "medium":
        verdicts.append("ACCEPTABLE - medium delay and medium distance")

    if not verdicts:
        verdicts.append("NO SPECIAL RULE TRIGGERED - route is acceptable")

    return verdicts


# ---------------------------------------------------------------------------
# Weighted Scoring
# ---------------------------------------------------------------------------

DELAY_SCORE = {"low": 1.0, "medium": 0.5, "high": 0.0}


def compute_route_score(
    route_info: dict,
    predicted_delay: str,
    time_of_day: str,
    max_cost: float,
) -> float:
    """
    Weighted score ∈ [0, 1].  Higher = better.
      cost_score:       how cheap the route is relative to the worst route
      delay_score:      1 if low, 0.5 if medium, 0 if high
      prob_score:       1 − P(high delay | time_of_day)
      restrict_score:   0 if passes restricted, else 1
    """
    cost_score = 1.0 - (route_info["total_cost"] / max_cost) if max_cost else 1.0
    d_score = DELAY_SCORE.get(predicted_delay, 0.5)
    prob_high = delay_probability(time_of_day, "high")
    prob_score = 1.0 - prob_high
    restrict_score = 0.0 if route_info["passes_restricted"] else 1.0

    score = (
        W_COST * cost_score
        + W_DELAY * d_score
        + W_PROB * prob_score
        + W_RESTRICT * restrict_score
    )
    return round(score, 4)


# ---------------------------------------------------------------------------
# Final Decision
# ---------------------------------------------------------------------------

def evaluate_routes(
    routes: Dict[str, dict],
    predictions: Dict[str, str],
    time_of_day: str,
) -> Tuple[str, Dict[str, dict]]:
    """
    Run rules + scoring on every candidate route and pick the best one.

    Returns:
        best_route_name, evaluation_details
    where evaluation_details maps route name → {
        predicted_delay, verdicts, score, rejected
    }
    """
    max_cost = max(r["total_cost"] for r in routes.values()) if routes else 1.0

    evaluation: Dict[str, dict] = {}

    for name, info in routes.items():
        predicted_delay = predictions[name]
        verdicts = apply_rules(name, info, predicted_delay)
        score = compute_route_score(info, predicted_delay, time_of_day, max_cost)
        rejected = any(v.startswith("REJECT") for v in verdicts)

        evaluation[name] = {
            "predicted_delay": predicted_delay,
            "verdicts": verdicts,
            "score": score,
            "rejected": rejected,
        }

    # Pick the non-rejected route with the highest score
    candidates = {n: e for n, e in evaluation.items() if not e["rejected"]}
    if candidates:
        best = max(candidates, key=lambda n: candidates[n]["score"])
    else:
        # All rejected — pick the least-bad one
        best = max(evaluation, key=lambda n: evaluation[n]["score"])

    return best, evaluation


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_routes = {
        "Route A": {
            "total_cost": 14.0,
            "distance_category": "short",
            "passes_restricted": False,
            "zones_visited": {"normal": 7},
            "congestion_summary": {"low": 5, "medium": 2},
        },
        "Route B": {
            "total_cost": 22.0,
            "distance_category": "long",
            "passes_restricted": True,
            "zones_visited": {"normal": 5, "restricted": 3},
            "congestion_summary": {"low": 3, "high": 5},
        },
    }
    preds = {"Route A": "low", "Route B": "high"}
    best, details = evaluate_routes(sample_routes, preds, "evening")
    for n, d in details.items():
        print(f"{n}: score={d['score']}, rejected={d['rejected']}, delay={d['predicted_delay']}")
        for v in d["verdicts"]:
            print(f"    {v}")
    print(f"\n>>> Best route: {best}")
