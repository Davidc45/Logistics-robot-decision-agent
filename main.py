"""
Logistics Robot Decision Agent – Main Entry Point

Runs the full pipeline:
  1. Search   → generate candidate routes on the warehouse grid
  2. ML       → predict delay level for each route
  3. Reasoning→ apply rules + probability scoring → select best route

Usage:
    python main.py
"""

from search import build_scenario_warehouse, generate_candidate_routes
from ml import DelayPredictor
from reasoning import evaluate_routes, delay_probability


def divider(title: str):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def run_scenario(scenario_name: str, time_of_day: str, predictor: DelayPredictor):
    """Execute one full decision-making scenario."""
    divider(f"SCENARIO: {scenario_name}  (time = {time_of_day})")

    # ------------------------------------------------------------------
    # 1. SEARCH – generate candidate routes
    # ------------------------------------------------------------------
    wh, start, goal = build_scenario_warehouse(scenario_name)

    print("\n[1] WAREHOUSE MAP")
    print("    S=Start  G=Goal  #=Wall  R=Restricted  T=High-traffic")
    print("    !=High-congestion  ~=Medium-congestion  .=Normal\n")
    print(wh.display(start=start, goal=goal))

    routes = generate_candidate_routes(wh, start, goal, num_routes=3)

    print(f"\n[2] CANDIDATE ROUTES FOUND: {len(routes)}")
    for name, info in routes.items():
        print(f"\n  --- {name} ---")
        print(f"    Steps:             {info['steps']}")
        print(f"    Total cost:        {info['total_cost']}")
        print(f"    Distance category: {info['distance_category']}")
        print(f"    Passes restricted: {info['passes_restricted']}")
        print(f"    Zones visited:     {info['zones_visited']}")
        print(f"    Congestion:        {info['congestion_summary']}")

    # ------------------------------------------------------------------
    # 2. ML – predict delay for each route
    # ------------------------------------------------------------------
    predictions = {}
    print("\n[3] ML DELAY PREDICTIONS")
    for name, info in routes.items():
        pred = predictor.predict_for_route(info, time_of_day)
        predictions[name] = pred
        print(f"    {name}: predicted delay = {pred}")

    # ------------------------------------------------------------------
    # 3. REASONING – rules + probability + final decision
    # ------------------------------------------------------------------
    best_name, evaluation = evaluate_routes(routes, predictions, time_of_day)

    print("\n[4] RULE EVALUATION & SCORING")
    for name, detail in evaluation.items():
        status = "REJECTED" if detail["rejected"] else "ACCEPTED"
        print(f"\n  --- {name} [{status}] ---")
        print(f"    Predicted delay: {detail['predicted_delay']}")
        print(f"    Score:           {detail['score']}")
        for v in detail["verdicts"]:
            print(f"      • {v}")

    # Probability context
    print(f"\n[5] PROBABILITY CONTEXT (time_of_day = {time_of_day})")
    print(f"    P(delay=low  | {time_of_day}) = {delay_probability(time_of_day, 'low')}")
    print(f"    P(delay=med  | {time_of_day}) = {delay_probability(time_of_day, 'medium')}")
    print(f"    P(delay=high | {time_of_day}) = {delay_probability(time_of_day, 'high')}")

    # Final decision
    divider("FINAL DECISION")
    best_info = routes[best_name]
    best_eval = evaluation[best_name]
    print(f"  Selected route : {best_name}")
    print(f"  Steps          : {best_info['steps']}")
    print(f"  Total cost     : {best_info['total_cost']}")
    print(f"  Predicted delay: {best_eval['predicted_delay']}")
    print(f"  Score          : {best_eval['score']}")
    print(f"  Reason         : ", end="")
    for v in best_eval["verdicts"]:
        print(v)

    # Show the map with all routes drawn
    print("\n[6] MAP WITH ROUTES")
    path_dict = {}
    for name, info in routes.items():
        label = name.split("(")[0].strip().replace("Route ", "")
        path_dict[label] = info["path"]
    print(wh.display(paths=path_dict, start=start, goal=goal))

    print()


def main():
    print("*" * 64)
    print("  LOGISTICS ROBOT DECISION AGENT")
    print("  AI Components: Search + ML + Rules + Probability")
    print("*" * 64)

    predictor = DelayPredictor()
    print(f"\nML model trained on {len(predictor.data)} samples.\n")

    # Run three different scenarios for the live demo
    run_scenario("default",       "morning", predictor)
    run_scenario("evening_rush",  "evening", predictor)
    run_scenario("night_quiet",   "night",   predictor)


if __name__ == "__main__":
    main()
