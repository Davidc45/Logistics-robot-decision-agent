"""
Search Module – Student 1 (Search Engineer)

Warehouse grid representation with zone metadata, plus BFS and A* algorithms
to generate multiple candidate routes from Start to Goal.
"""

import heapq
from collections import deque
from typing import List, Tuple, Dict, Optional

# ---------------------------------------------------------------------------
# Grid / Map Representation
# ---------------------------------------------------------------------------

# Zone types that appear in the dataset
ZONE_NORMAL = "normal"
ZONE_HIGH_TRAFFIC = "high_traffic"
ZONE_RESTRICTED = "restricted"

# Congestion levels
CONGESTION_LOW = "low"
CONGESTION_MEDIUM = "medium"
CONGESTION_HIGH = "high"


class WarehouseMap:
    """
    A grid-based warehouse where each cell has:
      - zone_type:        normal | high_traffic | restricted
      - congestion_level: low | medium | high
      - walkable:         whether the robot can physically pass through
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        # Default every cell to normal / low congestion / walkable
        self.grid = [
            [
                {
                    "zone_type": ZONE_NORMAL,
                    "congestion_level": CONGESTION_LOW,
                    "walkable": True,
                }
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]

    # ---- builders --------------------------------------------------------

    def set_zone(self, r: int, c: int, zone_type: str):
        self.grid[r][c]["zone_type"] = zone_type

    def set_congestion(self, r: int, c: int, level: str):
        self.grid[r][c]["congestion_level"] = level

    def set_wall(self, r: int, c: int):
        """Mark a cell as an impassable wall / obstacle."""
        self.grid[r][c]["walkable"] = False

    def cell(self, r: int, c: int) -> dict:
        return self.grid[r][c]

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def walkable(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.grid[r][c]["walkable"]

    def neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return 4-connected walkable neighbors."""
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.walkable(nr, nc):
                result.append((nr, nc))
        return result

    # ---- cost model ------------------------------------------------------

    def move_cost(self, r: int, c: int) -> float:
        """
        Movement cost INTO cell (r, c).
        Normal / low  = 1
        Congestion or zone penalties raise cost so A* naturally avoids them.
        """
        cell = self.grid[r][c]

        base = 1.0

        zone_penalty = {
            ZONE_NORMAL: 0.0,
            ZONE_HIGH_TRAFFIC: 2.0,
            ZONE_RESTRICTED: 5.0,
        }
        congestion_penalty = {
            CONGESTION_LOW: 0.0,
            CONGESTION_MEDIUM: 1.5,
            CONGESTION_HIGH: 3.0,
        }

        return (
            base
            + zone_penalty.get(cell["zone_type"], 0.0)
            + congestion_penalty.get(cell["congestion_level"], 0.0)
        )

    # ---- pretty-print ----------------------------------------------------

    def display(
        self,
        paths: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> str:
        """
        Return a string visualization of the warehouse grid.
        Paths is a dict like {"A": [(r,c),...], "B": [...]} – each path
        gets its label letter drawn on the grid.
        """
        path_symbols = ["A", "B", "C", "D", "E"]
        path_cells: Dict[Tuple[int, int], str] = {}
        if paths:
            for idx, (name, cells) in enumerate(paths.items()):
                sym = path_symbols[idx % len(path_symbols)]
                for cell in cells:
                    if cell != start and cell != goal:
                        path_cells[cell] = sym

        lines = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                if (r, c) == start:
                    row_chars.append(" S ")
                elif (r, c) == goal:
                    row_chars.append(" G ")
                elif (r, c) in path_cells:
                    row_chars.append(f" {path_cells[(r, c)]} ")
                elif not self.grid[r][c]["walkable"]:
                    row_chars.append(" # ")
                elif self.grid[r][c]["zone_type"] == ZONE_RESTRICTED:
                    row_chars.append(" R ")
                elif self.grid[r][c]["zone_type"] == ZONE_HIGH_TRAFFIC:
                    row_chars.append(" T ")
                elif self.grid[r][c]["congestion_level"] == CONGESTION_HIGH:
                    row_chars.append(" ! ")
                elif self.grid[r][c]["congestion_level"] == CONGESTION_MEDIUM:
                    row_chars.append(" ~ ")
                else:
                    row_chars.append(" . ")
            lines.append("|" + "|".join(row_chars) + "|")
        separator = "+" + "+".join(["---"] * self.cols) + "+"
        full = [separator]
        for line in lines:
            full.append(line)
            full.append(separator)
        return "\n".join(full)


# ---------------------------------------------------------------------------
# Search Algorithms
# ---------------------------------------------------------------------------

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs(
    warehouse: WarehouseMap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    Breadth-First Search – finds the path with the fewest hops (ignoring
    zone/congestion cost).  Useful as a baseline shortest-hop path.
    """
    queue = deque()
    queue.append((start, [start]))
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for nr, nc in warehouse.neighbors(r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None


def astar(
    warehouse: WarehouseMap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    extra_weight: Optional[Dict[Tuple[int, int], float]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """
    A* Search – finds the lowest-cost path using the warehouse move_cost
    model plus an optional extra_weight overlay (used to generate diverse
    alternative paths by penalizing cells on already-found routes).
    """
    open_set: list = []
    heapq.heappush(open_set, (0.0, start))
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            return path[::-1]

        for nr, nc in warehouse.neighbors(*current):
            neighbor = (nr, nc)
            cost = warehouse.move_cost(nr, nc)
            if extra_weight and neighbor in extra_weight:
                cost += extra_weight[neighbor]
            tentative = g_score[current] + cost
            if tentative < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative
                f = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
                came_from[neighbor] = current

    return None


# ---------------------------------------------------------------------------
# Candidate Route Generator
# ---------------------------------------------------------------------------

def generate_candidate_routes(
    warehouse: WarehouseMap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    num_routes: int = 3,
) -> Dict[str, dict]:
    """
    Produce multiple candidate routes using different strategies:
      Route A – A* optimal (considers zone/congestion cost)
      Route B – BFS shortest-hop (ignores cost, just fewest steps)
      Route C+ – A* with penalty on prior paths (diverse alternatives)

    Returns a dict keyed by route name, each value is:
      {
        "path": [(r,c), ...],
        "steps": int,
        "total_cost": float,
        "zones_visited": {zone_type: count},
        "congestion_summary": {level: count},
        "passes_restricted": bool,
        "distance_category": "short" | "medium" | "long",
      }
    """
    routes: Dict[str, dict] = {}
    used_paths: List[List[Tuple[int, int]]] = []

    # ---- Route A: A* optimal cost path -----------------------------------
    path_a = astar(warehouse, start, goal)
    if path_a:
        routes["Route A (A* cost-optimal)"] = _analyze_path(warehouse, path_a)
        used_paths.append(path_a)

    # ---- Route B: BFS shortest hop path ----------------------------------
    path_b = bfs(warehouse, start, goal)
    if path_b:
        if path_b != path_a:
            routes["Route B (BFS shortest-hop)"] = _analyze_path(warehouse, path_b)
            used_paths.append(path_b)
        else:
            # BFS matched A*; generate a penalty-based alternative as B
            penalty_b: Dict[Tuple[int, int], float] = {}
            for cell in path_a:
                penalty_b[cell] = penalty_b.get(cell, 0.0) + 8.0
            alt_b = astar(warehouse, start, goal, extra_weight=penalty_b)
            if alt_b and alt_b != path_a:
                routes["Route B (A* alternate)"] = _analyze_path(warehouse, alt_b)
                used_paths.append(alt_b)

    # ---- Route C+: diverse A* alternatives --------------------------------
    route_letter = ord("C")
    attempts = 0
    while len(routes) < num_routes and attempts < 10:
        attempts += 1
        penalty: Dict[Tuple[int, int], float] = {}
        for prev_path in used_paths:
            for cell in prev_path:
                penalty[cell] = penalty.get(cell, 0.0) + 6.0 * attempts

        alt = astar(warehouse, start, goal, extra_weight=penalty)
        if alt is None or alt in used_paths:
            break
        label = chr(route_letter)
        routes[f"Route {label} (A* alternative)"] = _analyze_path(warehouse, alt)
        used_paths.append(alt)
        route_letter += 1

    return routes


def _analyze_path(
    warehouse: WarehouseMap, path: List[Tuple[int, int]]
) -> dict:
    """Compute summary statistics for a single path."""
    total_cost = 0.0
    zones: Dict[str, int] = {}
    congestion: Dict[str, int] = {}
    passes_restricted = False

    for r, c in path:
        total_cost += warehouse.move_cost(r, c)
        zt = warehouse.cell(r, c)["zone_type"]
        cl = warehouse.cell(r, c)["congestion_level"]
        zones[zt] = zones.get(zt, 0) + 1
        congestion[cl] = congestion.get(cl, 0) + 1
        if zt == ZONE_RESTRICTED:
            passes_restricted = True

    steps = len(path) - 1

    if steps <= 6:
        distance_category = "short"
    elif steps <= 12:
        distance_category = "medium"
    else:
        distance_category = "long"

    return {
        "path": path,
        "steps": steps,
        "total_cost": round(total_cost, 2),
        "zones_visited": zones,
        "congestion_summary": congestion,
        "passes_restricted": passes_restricted,
        "distance_category": distance_category,
    }


# ---------------------------------------------------------------------------
# Default Warehouse Builder
# ---------------------------------------------------------------------------

def build_default_warehouse() -> Tuple[WarehouseMap, Tuple[int, int], Tuple[int, int]]:
    """
    Build a 10x10 warehouse map designed so that BFS, A*, and penalty-A*
    each produce a genuinely different path.

    Key design: A wall barrier across the middle forces all traffic through
    one of three "corridors":
      - Central gap (col 5):  high-traffic / high-congestion — BFS uses
        this because it has the fewest hops.
      - Northern route:  wraps around the top — longer in hops but low
        cost, so A* prefers it.
      - Southern route:  goes through a restricted zone — short but
        gets rejected by reasoning rules.
    """
    wh = WarehouseMap(10, 10)

    # ---- Central wall barrier (row 4, most of the row) ----
    # Leave a gap at col 5 (the only direct crossing point)
    for c in range(0, 10):
        if c != 5:
            wh.set_wall(4, c)

    # ---- The central gap is high-traffic + high-congestion ----
    wh.set_zone(4, 5, ZONE_HIGH_TRAFFIC)
    wh.set_congestion(4, 5, CONGESTION_HIGH)
    wh.set_zone(3, 5, ZONE_HIGH_TRAFFIC)
    wh.set_congestion(3, 5, CONGESTION_HIGH)
    wh.set_zone(5, 5, ZONE_HIGH_TRAFFIC)
    wh.set_congestion(5, 5, CONGESTION_HIGH)

    # ---- Northern bypass walls (force the long detour) ----
    # Wall at row 2, cols 7-8  — makes the northern route wrap further
    wh.set_wall(2, 7)
    wh.set_wall(2, 8)

    # ---- Southern restricted zone (rows 6-7, cols 3-5) ----
    for r in range(6, 8):
        for c in range(3, 6):
            wh.set_zone(r, c, ZONE_RESTRICTED)
            wh.set_congestion(r, c, CONGESTION_LOW)

    # ---- Medium congestion in the northern corridor ----
    for c in range(6, 10):
        wh.set_congestion(0, c, CONGESTION_MEDIUM)
    wh.set_congestion(1, 9, CONGESTION_MEDIUM)

    # ---- Small obstacles for flavor ----
    wh.set_wall(7, 7)
    wh.set_wall(6, 8)

    # ---- Medium congestion in the southern approach ----
    wh.set_congestion(8, 6, CONGESTION_MEDIUM)
    wh.set_congestion(8, 7, CONGESTION_MEDIUM)

    start = (0, 0)
    goal = (9, 9)

    return wh, start, goal


def build_scenario_warehouse(
    scenario: str = "default",
) -> Tuple[WarehouseMap, Tuple[int, int], Tuple[int, int]]:
    """
    Build different warehouse configurations for demo scenarios.
    Scenarios: 'default', 'evening_rush', 'night_quiet'
    """
    if scenario == "evening_rush":
        wh, start, goal = build_default_warehouse()
        # Increase congestion everywhere during evening rush
        for r in range(wh.rows):
            for c in range(wh.cols):
                if wh.grid[r][c]["walkable"]:
                    current = wh.grid[r][c]["congestion_level"]
                    if current == CONGESTION_LOW:
                        wh.set_congestion(r, c, CONGESTION_MEDIUM)
                    elif current == CONGESTION_MEDIUM:
                        wh.set_congestion(r, c, CONGESTION_HIGH)
        return wh, start, goal

    elif scenario == "night_quiet":
        wh, start, goal = build_default_warehouse()
        # Reduce congestion at night
        for r in range(wh.rows):
            for c in range(wh.cols):
                if wh.grid[r][c]["walkable"]:
                    wh.set_congestion(r, c, CONGESTION_LOW)
        return wh, start, goal

    else:
        return build_default_warehouse()


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wh, start, goal = build_default_warehouse()

    print("=" * 60)
    print("  WAREHOUSE MAP  (S=Start, G=Goal, #=Wall, R=Restricted,")
    print("                   T=High-traffic, !=High-congestion, ~=Med)")
    print("=" * 60)
    print(wh.display(start=start, goal=goal))
    print()

    routes = generate_candidate_routes(wh, start, goal, num_routes=3)

    print("=" * 60)
    print("  CANDIDATE ROUTES")
    print("=" * 60)
    for name, info in routes.items():
        print(f"\n--- {name} ---")
        print(f"  Steps:             {info['steps']}")
        print(f"  Total cost:        {info['total_cost']}")
        print(f"  Distance category: {info['distance_category']}")
        print(f"  Passes restricted: {info['passes_restricted']}")
        print(f"  Zones visited:     {info['zones_visited']}")
        print(f"  Congestion:        {info['congestion_summary']}")

    print("\n" + "=" * 60)
    print("  MAP WITH ROUTES")
    print("=" * 60)
    path_dict = {}
    for name, info in routes.items():
        label = name.split("(")[0].strip().replace("Route ", "")
        path_dict[label] = info["path"]
    print(wh.display(paths=path_dict, start=start, goal=goal))
