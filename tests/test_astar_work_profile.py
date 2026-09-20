import numpy as np

from astar_work_profile import profile_astar_work


def test_profiled_astar_returns_path_and_mechanistic_counts():
    # 0 -- 1 -- 2
    indptr = np.asarray([0, 1, 3, 4], dtype=np.int32)
    indices = np.asarray([1, 0, 2, 1], dtype=np.int32)
    coords = np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    zeros_float = np.zeros(3, dtype=np.float32)
    zeros_int = np.zeros(3, dtype=np.int32)
    paths = np.asarray([[1, -1], [2, -1]], dtype=np.int32)
    result = profile_astar_work(
        indptr=indptr,
        indices=indices,
        coords=coords,
        jam_values=zeros_float,
        jam_timestamps=zeros_int,
        flow_values=zeros_float,
        flow_timestamps=zeros_int,
        path_values=paths,
        path_timestamps=np.zeros(2, dtype=np.int32),
        start=0,
        goal=2,
        cost_params={"max_aoi_ms": 100, "proximity_radius": 25},
        current_time_ms=0,
    )
    assert result.path == [0, 1, 2]
    assert result.graph_nodes_initialized == 3
    assert result.reservation_slots_scanned == 4
    assert result.valid_reservations_in_corridor == 2
    assert result.nodes_expanded >= 3
    assert result.edge_evaluations >= 3
    assert result.heap_pushes >= 3


def test_start_at_goal_has_no_index_or_search_work():
    result = profile_astar_work(
        indptr=np.asarray([0, 0], dtype=np.int32),
        indices=np.asarray([], dtype=np.int32),
        coords=np.asarray([[0, 0]], dtype=np.float32),
        jam_values=np.zeros(1, dtype=np.float32),
        jam_timestamps=np.zeros(1, dtype=np.int32),
        flow_values=np.zeros(1, dtype=np.float32),
        flow_timestamps=np.zeros(1, dtype=np.int32),
        path_values=np.asarray([[-1]], dtype=np.int32),
        path_timestamps=np.zeros(1, dtype=np.int32),
        start=0,
        goal=0,
        cost_params={},
        current_time_ms=0,
    )
    assert result.path == [0]
    assert result.edge_evaluations == 0
