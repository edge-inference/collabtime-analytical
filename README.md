# Analytical Performance Model

Analytical decomposition of centralized and DSM coordination costs, physical
capacity, queueing, and data freshness. Numerical conclusions are conditional
on the configured architecture and parameter scenario.

## Overview

This module implements mathematical models to compare:
- **Centralized**: Batch planning with broadcast distribution
- **DSM**: Distributed shared memory with halo gossip protocol

Key metrics analyzed:
- Information propagation time
- Total task latency 
- System stability boundaries
- Age of Information violations

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic analysis
python run.py

# Generate plots and detailed report
python run.py -v

# Custom configuration
python run.py -c my_config.yaml -o my_results/
```

### Thesis analytical studies

`config_thesis_strong.yaml` contains the shared warehouse geometry and nominal
service assumptions for the 120x100 layout. Arrival rates are expressed
internally in tasks/ms; the configured grid corresponds to 0.2--6.0 tasks/s.

The first five commands generate capacity-boundary and sensitivity analyses.
The final command is the separate strict strong-scaling study at a fixed total
arrival rate.

```bash
venv/bin/python run.py -c config_thesis_strong.yaml -o results/thesis_strong
venv/bin/python plot_analytical_summary.py
venv/bin/python plot_data_plane_sensitivity.py
venv/bin/python sweep_scheduler_sensitivity.py
venv/bin/python sweep_propagation_sensitivity.py
venv/bin/python run_strong_scaling_fixed.py --arrival-rate 5.0
```

### Scheduler service-demand pilot and full protocol

The scheduler-capacity sensitivity uses conditional service-demand parameters.
The current event-driven script is a component microbenchmark, not a complete
per-task calibration. Read `SCHEDULER_CAPACITY_STUDY.md` for the required
operation-count, resource-demand, queueing, and validation protocol before
using scheduler-capacity results in a thesis figure. To reproduce the pilot,
build the warehouse Cython A* extension and run:

```bash
cd /home/modfi/ivalab/extern/warehouse
/home/modfi/ivalab/venv/bin/python perf/setup_cython.py build_ext --inplace

cd /home/modfi/models/research/analytical
/home/modfi/ivalab/venv/bin/python scheduler_demand_study.py
```

The pilot reports isolated elapsed operation costs in `ms/operation`, an assumed
one-assignment/one-path aggregate in `ms/task`, and saturated throughput in
`tasks/s`. It keeps network latency separate, compares candidate scaling laws,
bootstraps the linear parameters, and checks capacity with threads and
independent worker processes. Outputs are written to
`results/scheduler_demand_study/`; they are not thesis-ready calibration data.

The complete scheduler-capacity workflow is split into reproducible stages:

```bash
# Implementation operation rates and visit ratios
/home/modfi/ivalab/venv/bin/python warehouse_scheduler_trace.py

# Component CPU/elapsed cost versus algorithmic work factors
/home/modfi/ivalab/venv/bin/python scheduler_component_factorial.py --cpu-affinity 5

# Trace decomposition and comparison with the configured analytical scenario
venv/bin/python analyze_scheduler_trace.py

# Open-loop queue and worker-scaling validation
/home/modfi/ivalab/venv/bin/python scheduler_open_loop_validation.py
```

These stages intentionally distinguish the implemented polling baseline, the
event-driven full-reservation counterfactual, and any batched global planner.
CPU time calibrates CPU service demand on the disclosed host; elapsed service
time, queue wait, and end-to-end response remain separate quantities. A CPU
coefficient is not treated as portable across machines or planner algorithms.

Analytical figure outputs are written under `results/thesis_strong/`; scheduler
calibration and validation outputs are under
`results/scheduler_capacity_study/`. Both are intentionally ignored by Git.

## Key Models

### Propagation Time

**Centralized:**
```
T_prop_cent = Δ_B/2 + T_solve(N) + D_tree × (δ_hop + δ_ser) + T_ack,cent
```
- Δ_B/2: Expected wait for uniformly phased arrivals within a batch period
- T_solve(N): Solver complexity (a×N² + b×N + c)
- D_tree: Broadcast tree depth
- δ_hop: Network hop delay (10ms default)
- δ_ser: Per-hop serialization assumption
- T_ack,cent: Optional propagation acknowledgment RTT (zero by default)

**DSM:**
```
T_prop_dsm = T_g/2 + (dist_hops + ceil(log_f(N))) × (δ_hop + δ_ser) + T_ack,dsm
```
- T_g/2: Expected wait for a uniformly phased update within a gossip period
- dist_hops: Distance between tile owners (2 hops default)
- f: Gossip fanout (2 default)
- N: Fleet size
- T_ack,dsm: Optional propagation acknowledgment RTT (zero by default)

### Total Latency

```
T_total = T_prop + T_claim + T_queue + T_travel + T_handle
```

Where:
- T_prop: Information propagation (above)
- T_claim: Task claiming overhead (contention in DSM)
- T_queue: Path congestion delays (G/G/1 and G/G/c approximations)
- T_travel: path_cells × t_traverse (ms), with t_traverse = cell_length / robot_speed
- T_handle: Task execution time at nodes (work_time)

### Stability Analysis

Fleet capacity with coordination overhead:
```
C = N × μ_eff / (1 + overhead × λ)
```

System stable when λ < C × ρ_max (ρ_max = 0.85 default).

## Configuration

Edit `config.yaml` to adjust:

### System Parameters
```yaml
system:
  fleet_sizes: [4, 8, 12, 16, 20, 24, 32, 48, 64]
  warehouse:
    width: 20            # cells (dimensionless)
    height: 15           # cells (dimensionless)
    cell_length: 1.0     # meters per cell (longitudinal)
    aisle_width: 2.0     # meters (lateral; capacity-related)
  edge_capacity: 2       # lanes per edge (system-level capacity)
  node_capacity: 1       # bays per node (system-level capacity)
  robot:
    speed: 0.001         # m/ms
    work_time: 5000.0    # ms per task at nodes
```

### Network Parameters
```yaml
network:
  hop_delay: 10.0  # milliseconds
  central:
    batch_period: 200.0
    solver_complexity: {a: 0.001, b: 2.0, c: 10.0}
  dsm:
    gossip_fanout: 2
    gossip_period: 150.0
```

### Analysis Settings
```yaml
analysis:
  metrics: ["propagation_time", "total_latency", "stability_boundary"]
  percentiles: [50, 90, 95, 99]
```

## Interpretation Rules

- A propagation threshold is the first sampled fleet size at which one
  configured delay expression is lower than the other. It is not an empirical
  warehouse crossover.
- A scheduler-capacity threshold is conditional on scheduler service demand,
  operation rates, and effective worker capacity. Uncalibrated values are
  sensitivity scenarios, not measured boundaries.
- The 300 ms AoI target is an illustrative freshness budget for fast-changing
  local state. It is not a universal warehouse networking requirement.
- Analytical results motivate simulation or implementation experiments; they
  do not predetermine which architecture must win.

## Files

- `config.yaml` - Analysis configuration
- `models.py` - Core mathematical models
- `comparison.py` - DSM vs centralized analysis
- `visualize.py` - Plot generation
- `run.py` - Main analysis runner

## Output

### Text Report
```
=== DSM vs Centralized Performance Analysis ===

First sampled propagation-delay advantage: 4 robots
Scheduler-bottleneck threshold: not reached in the analyzed fleet range

Performance Advantages:
  First sampled propagation advantage: 4 robots
  Max throughput improvement: 1.00x
  Average latency improvement: 1.001x (11 jointly stable points)
  Fleet sizes with scheduler-limited centralized capacity: 0

Interpretation:
  The scheduler is not capacity-limiting in the analyzed range
```

### Visualizations
- `propagation_vs_fleet_size.png` - Shows crossover point
- `stability_boundaries.png` - Throughput capacity comparison  
- `latency_heatmap.png` - Performance across operating conditions
- `aoi_violations.png` - Information freshness analysis
- `crossover_analysis.png` - Detailed crossover characteristics

## Interpretation

Treat any architectural advantage as conditional until it survives parameter
sensitivity and an independent implementation or simulation evaluation. If no
threshold occurs in the evaluated domain, report that directly instead of
substituting the last sampled fleet size.

## Parameter Sensitivity

The analysis includes sensitivity studies for:
- Combined per-hop communication delay (1-50ms)
- Central batch and DSM gossip periods
- Solver complexity coefficients
- Gossip fanout (1, 2, 4, 8)
- Conflict probability (0.05-0.3)

Critical parameters for DSM advantage:
1. **Solver complexity**: Higher quadratic term favors DSM
2. **Network delay**: Lower latency helps DSM gossip
3. **Gossip fanout**: Higher fanout improves DSM propagation
4. **Fleet size**: DSM scales better than centralized

## Modeling Assumptions

- Spatial model is a grid of cells (dimensionless). Physical length per cell is `cell_length` (meters).
- Robot traversal time per cell is derived: `t_traverse = cell_length / robot_speed` (ms). This is the single source of truth for movement time in the analytical model.
- Queueing uses G/G/1 and G/G/c approximations with variance parameters (`arrival_cv_squared`, `service_cv_squared`).
- Stability follows a utilization cap `ρ_max` applied to effective capacity; max stable λ is computed per fleet size including coordination overhead.
- Communication:
  - Centralized and DSM periodic waits use half the configured period for mean-delay analysis.
  - Hop and serialization delay enter additively per logical hop.
  - Propagation acknowledgment RTTs are architecture-specific and zero unless an explicit request/ack exchange is modeled.
- Periodic-update AoI uses the exact fixed-period, fixed-delay sawtooth model.
  The separate Poisson helper states its exponential-age assumption explicitly.

These are tractable modeling assumptions whose sensitivity and implementation
mapping must be reported with each result.

## Integration with Warehouse DSM

Results feed into:
- Mesa simulation parameters (warehouse/experiments/configs.yaml)
- Lingua Franca timing constraints (warehouse/lf/dsm_coordinator.lf)
- DSM configuration (warehouse/dsm/api.py parameters)

Use conditional analytical thresholds to choose informative experimental fleet
sizes on both sides of the modeled boundary, then test whether the predicted
mechanism appears in the warehouse implementation.
