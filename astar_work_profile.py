"""Untimed work counters for the warehouse Cython A* implementation."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AStarWork:
    path: list[int]
    graph_nodes_initialized: int
    reservation_slots_scanned: int
    valid_reservations_in_corridor: int
    nodes_expanded: int
    edge_evaluations: int
    heap_pushes: int


def profile_astar_work(
    *,
    indptr: np.ndarray,
    indices: np.ndarray,
    coords: np.ndarray,
    jam_values: np.ndarray,
    jam_timestamps: np.ndarray,
    flow_values: np.ndarray,
    flow_timestamps: np.ndarray,
    path_values: np.ndarray,
    path_timestamps: np.ndarray,
    start: int,
    goal: int,
    cost_params: dict[str, float],
    current_time_ms: int,
) -> AStarWork:
    """Mirror `perf.astar_fast` and return operation counts, not timings."""
    num_nodes = int(coords.shape[0])
    reservation_slots = int(path_values.size)
    if start == goal:
        return AStarWork([start], 0, 0, 0, 0, 0, 0)

    alpha = float(cost_params.get("alpha", 2.0))
    beta = float(cost_params.get("beta", 0.5))
    max_aoi_ms = int(cost_params.get("max_aoi_ms", 5000))
    conflict_penalty = float(cost_params.get("conflict_penalty", 100.0))
    proximity_radius = float(cost_params.get("proximity_radius", 25.0))

    conflict_counts = np.zeros(num_nodes, dtype=np.int32)
    start_x, start_y = float(coords[start, 0]), float(coords[start, 1])
    goal_x, goal_y = float(coords[goal, 0]), float(coords[goal, 1])
    min_x = min(start_x, goal_x) - proximity_radius
    max_x = max(start_x, goal_x) + proximity_radius
    min_y = min(start_y, goal_y) - proximity_radius
    max_y = max(start_y, goal_y) + proximity_radius
    valid_reservations = 0
    for agent_index in range(path_values.shape[0]):
        if current_time_ms - int(path_timestamps[agent_index]) > max_aoi_ms:
            continue
        for path_index in range(path_values.shape[1]):
            node = int(path_values[agent_index, path_index])
            if node < 0 or node >= num_nodes:
                continue
            node_x, node_y = float(coords[node, 0]), float(coords[node, 1])
            if min_x <= node_x <= max_x and min_y <= node_y <= max_y:
                conflict_counts[node] += 1
                valid_reservations += 1

    open_set: list[tuple[float, int]] = []
    heapq.heappush(open_set, (0.0, start))
    came_from: dict[int, int] = {}
    g_score: dict[int, float] = {start: 0.0}
    nodes_expanded = 0
    edge_evaluations = 0
    heap_pushes = 1

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.insert(0, current)
            return AStarWork(
                path,
                num_nodes,
                reservation_slots,
                valid_reservations,
                nodes_expanded,
                edge_evaluations,
                heap_pushes,
            )

        current_g = g_score.get(current, math.inf)
        for edge_index in range(int(indptr[current]), int(indptr[current + 1])):
            edge_evaluations += 1
            neighbor = int(indices[edge_index])
            cost = 1.0
            if current_time_ms - int(jam_timestamps[neighbor]) <= max_aoi_ms:
                cost += alpha * float(jam_values[neighbor])
            if current_time_ms - int(flow_timestamps[neighbor]) <= max_aoi_ms:
                cost += beta * float(flow_values[neighbor])
            if conflict_counts[neighbor] > 0:
                cost += conflict_penalty

            tentative_g = current_g + cost
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heuristic = abs(float(coords[goal, 0]) - float(coords[neighbor, 0]))
                heuristic += abs(float(coords[goal, 1]) - float(coords[neighbor, 1]))
                heapq.heappush(open_set, (tentative_g + heuristic, neighbor))
                heap_pushes += 1

    return AStarWork(
        [],
        num_nodes,
        reservation_slots,
        valid_reservations,
        nodes_expanded,
        edge_evaluations,
        heap_pushes,
    )


def profile_workload_path(workload: Any, start: int, goal: int) -> AStarWork:
    return profile_astar_work(
        indptr=workload.indptr,
        indices=workload.indices,
        coords=workload.coords,
        jam_values=workload.jam_values,
        jam_timestamps=workload.jam_timestamps,
        flow_values=workload.flow_values,
        flow_timestamps=workload.flow_timestamps,
        path_values=workload.path_values,
        path_timestamps=workload.path_timestamps,
        start=start,
        goal=goal,
        cost_params=workload.cost_params,
        current_time_ms=0,
    )
