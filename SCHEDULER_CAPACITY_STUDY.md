# Scheduler Capacity Study Protocol

## Status

The existing `scheduler_demand_study.py` output is a pilot component
microbenchmark. It is not a complete calibration of scheduler demand per
completed task and must not be used as thesis evidence by itself. In
particular, the pilot assumes one event-triggered assignment and one path
request per task, while the warehouse implementation polls for assignments and
can replan paths.

This protocol defines the evidence required before the centralized scheduler
capacity term is used in a thesis figure.

## Research Questions

1. How does the cost of each centralized scheduler operation vary with fleet
   size, task backlog, map size, path length, reservation horizon, and robot
   density?
2. How many visits to each scheduler operation occur per completed task under
   each workload and coordination implementation?
3. What CPU, worker-occupancy, and end-to-end response demands do those visits
   produce?
4. At what offered load does the scheduler become unstable, and how accurately
   do the analytical capacity and queueing models predict that boundary?
5. Which conclusions survive changes in hardware, worker count, workload,
   warehouse geometry, and algorithmic policy?

## Systems That Must Not Be Conflated

The study reports these systems separately:

1. **Implemented polling baseline.** Every idle centralized robot invokes the
   assignment method once per simulator step. The current A* wrapper receives
   a dummy `1 x 1` reservation array. Mesa activates robots sequentially, so a
   semaphore configured with multiple replicas does not create concurrent
   scheduler execution in the warehouse simulation.
2. **Event-driven per-robot planner.** A task event causes assignment work and
   an individual route request. A fleet-wide `N x h` reservation snapshot is
   visible to route planning. This is an explicit counterfactual design until
   implemented end to end.
3. **Batched global planner.** A planning cycle jointly updates paths or task
   assignments for multiple robots. Its cost is per planning cycle and must be
   amortized over the tasks or decisions produced by that cycle. It cannot be
   calibrated from a single-agent A* call.

Any thesis equation or figure must identify which of these systems it models.

## Current Model Audit

The existing analytical and warehouse implementations do not yet describe one
common scheduler:

- Chapter 3 places an allocator runtime `T_solve(N)` inside centralized
  propagation latency, while scheduler capacity uses a separately configured
  `D_sched(N)`. These can represent cycle latency and resource demand,
  respectively, but only if their relationship, batching policy, and
  amortization are defined. At present their coefficients are independent.
- The warehouse baseline performs greedy per-agent assignment and individual
  Cython A* calls. It does not implement the quadratic batched allocator used by
  `T_solve(N)`.
- `scheduler_replicas = 10` enters the analytical capacity as ten independent
  workers. Warehouse agents are activated sequentially, and the semaphore does
  not produce concurrent service. Thread and process experiments must therefore
  be reported as separate execution models.
- The configured stability boundary multiplies capacities by an 85% utilization
  target. That is an engineering operating limit, not the mathematical
  stability condition `rho < 1`; figures and prose must distinguish them.
- Total task response currently adds propagation, claim, travel, and node time
  but no scheduler queue wait. If scheduler saturation is used to explain a
  throughput plateau, its queueing delay must also appear in latency analysis.
- Under polling, scheduler work is not a function of fleet size alone. It also
  depends on idle population, poll period, offered task rate, failures, and
  replanning. The appropriate load expression is `sum_k lambda_k S_k`, not only
  `lambda_task D_sched(N)`.

These are hypotheses to reconcile before changing thesis figures; they are not
resolved by choosing new numerical coefficients.

## Operational Model

For scheduler operation type `k` and resource `j`, define the visit ratio

```text
V_k = completed operation-k visits / completed tasks
```

over a stated steady observation interval. Define resource demand as

```text
D_j = sum_k V_k S_jk,
```

where `S_jk` is busy service time at resource `j` for operation `k`. The
utilization law is

```text
U_j = X D_j,
```

where `X` is completed-task throughput. This identity is validated directly
from measured busy time, completions, and utilization.

CPU and worker occupancy are separate service centers. For background
operation rates `b_k(N)` and task visit ratios `V_k`, their offered loads are

```text
L_cpu    = sum_k [b_k(N) + lambda_task V_k] S_cpu,k
L_worker = sum_k [b_k(N) + lambda_task V_k] S_elapsed,k.
```

The admissible task rate must satisfy both `L_cpu < C_cpu` and
`L_worker < C_worker,eff`, as well as any network or storage constraints.
`C_worker,eff` comes from the stated execution model or a measured scaling
curve; it is not set equal to the configured thread count without evidence.
Thus CPU time is a resource demand, not a substitute for elapsed service or
end-to-end response time.

The centralized workload includes at least:

- task lookup and assignment;
- claim/registry mutation;
- route planning;
- reservation construction, commit, and release;
- position, flow, and congestion-state updates;
- retries, replans, and idle/staging routes.

Calls caused by polling are reported as a separate workload class rather than
silently normalized into an event-driven task cost.

## Timing Quantities

Each operation records three distinct quantities:

1. **CPU demand:** thread or process CPU time consumed by the operation. This
   calibrates a CPU service center and is hardware-specific.
2. **In-service elapsed time:** wall time after a worker starts service and
   before it releases the worker. It includes synchronous lock or I/O stalls
   but excludes queue wait.
3. **Response time:** queue wait plus in-service elapsed time, with network and
   external storage delays reported as separate service centers when present.

CPU demand is not presented as end-to-end latency. Isolated wall time is not
called CPU demand. Parallel requests are not summed using elapsed time because
their intervals can overlap.

## Algorithmic Cost Structure

The model form follows the implementation rather than selecting a polynomial
from fleet size alone.

For the current greedy assignment scan,

```text
S_assign = f(N_active, Q_scan),
```

with expected work linear in the assigned-agent set and bounded task scan.

For the Cython A* implementation, conflict-index construction performs work
proportional to the graph size and the reservation input:

```text
S_index = f(|V|, N_active h).
```

Path search additionally depends on graph edges, route geometry, congestion,
and the number of expanded states. Consequently, an apparent `D0 + alpha N`
fit is valid only for fixed geometry, fixed reservation horizon, and a fixed
route/workload distribution. General claims must retain those covariates.

For a batched global planner, define cycle demand `S_cycle(N, B, h, ...)` and
the number of useful decisions `K_cycle`; its task-normalized demand is
`S_cycle / K_cycle`. It is not interchangeable with per-robot A* demand.

## Experimental Design

### Phase A: Component Characterization

Measure assignment, claim, path planning, reservation, and state-update
operations across:

- fleet sizes `N = {50, 100, 200, 300, 500, 600, 700, 800}`;
- fixed and scaled warehouse geometries;
- task backlog and assignment scan limits;
- reservation horizons and active-fleet fractions;
- route-distance and congestion strata;
- cold and warmed execution states.

Record CPU time, elapsed time, path outcome, path length, and all available
algorithmic work counters. Randomize scenario order within replicate and use
paired route/task samples where comparisons require common random numbers.

### Phase B: Visit-Ratio Measurement

Instrument the implemented scheduler during headless warehouse runs. Record
operation counts and busy time by method, completed/claimed/failed tasks,
replans, retries, idle polls, and state updates. Evaluate both fixed offered
load and load proportional to fleet size. Report visit ratios with confidence
intervals; do not assume one path request per task.

### Phase C: Capacity and Queue Validation

Drive an isolated scheduler service with an open-loop arrival process that is
independent of completions. Sweep offered load below and above predicted
capacity. For each worker count, report:

- achieved throughput and backlog growth;
- aggregate CPU utilization and worker occupancy;
- p50, p95, and p99 queue, service, and response times;
- service-time squared coefficient of variation;
- prediction error for utilization, mean wait, and saturation throughput.

Thread and process workers are separate execution models. A worker-count
parameter is not assumed to scale linearly; measured contention/coherency loss
is modeled explicitly or workers are placed on independently specified hosts.

### Phase D: Sensitivity and External Validity

Propagate uncertainty in visit ratios and service demands through the capacity
model. Report a range over hardware calibration, worker efficiency, workload,
and planning policy. Use literature results only to check orders of magnitude
and algorithm classes, not to transplant runtime constants between machines.

## Statistical Analysis

- Treat process executions or independently randomized workloads as the unit
  of replication; inner timing iterations are subsamples.
- Use nested or cluster bootstrap intervals that preserve within-execution
  dependence.
- Report effect sizes and confidence intervals, not only fitted coefficients or
  p-values.
- Select models using blocked out-of-scenario prediction, with algorithmic
  structure preferred over marginal improvements from higher-order curves.
- Reserve complete fleet sizes or workload regimes for out-of-sample
  validation.
- Record source revisions, compiler flags, Python/Cython versions, CPU model,
  core count, frequency policy, affinity, and background-load controls.

## Acceptance Criteria

The scheduler-capacity result is thesis-ready only when:

1. the modeled scheduler architecture matches the implementation or is clearly
   identified as a counterfactual scenario;
2. operation visit ratios are measured rather than assumed;
3. CPU demand, service elapsed time, queue wait, and network delay are not
   conflated;
4. utilization-law error is within a stated tolerance on held-out runs;
5. the capacity boundary is validated by sustained backlog behavior under
   open-loop load;
6. uncertainty and sensitivity do not reverse the stated conclusion without
   that dependence being shown explicitly;
7. all scripts, configurations, raw observations, and environment metadata are
   reproducible.

## Methodological Basis

- P. J. Denning and J. P. Buzen, "The Operational Analysis of Queueing Network
  Models," *ACM Computing Surveys*, 1978.
- W. Whitt, "The Queueing Network Analyzer," *Bell System Technical Journal*,
  1983.
- T. Kalibera and R. Jones, "Rigorous Benchmarking in Reasonable Time," ISMM,
  2013.
- N. J. Gunther, "A General Theory of Computational Scalability Based on
  Rational Functions," 2008.
- J. Li et al., "Lifelong Multi-Agent Path Finding in Large-Scale Warehouses,"
  AAAI, 2021.
- Z. Chen et al., "Traffic Flow Optimisation for Lifelong Multi-Agent Path
  Finding," AAAI, 2024.
- H. Ma et al., "Lifelong Path Planning with Kinematic Constraints for
  Multi-Agent Pickup and Delivery," AAAI, 2019.

## Literature Triangulation

Published runtime values establish algorithm classes and experimental practice,
not portable constants:

- Denning and Buzen define service demand operationally from measured resource
  busy time and completions, yielding the utilization identity `U = X D`.
  https://www.columbia.edu/~ww2040/8100S12/DenningBuzen1978.pdf
- Whitt's Queueing Network Analyzer treats multiserver nodes with general
  interarrival and service variability through measured moments and explicit
  approximation assumptions.
  https://www.columbia.edu/~ww2040/QNA_1983.pdf
- Kalibera and Jones require repetition at the levels where nondeterminism
  enters and recommend effect-size confidence intervals rather than isolated
  inner-loop timings.
  https://doi.org/10.1145/2464157.2464160
- Li et al. separate task assignment from lifelong path planning and use a
  bounded rolling horizon to make centralized replanning practical for up to
  1,000 agents.
  https://doi.org/10.1609/aaai.v35i13.17344
- Chen et al. evaluate 24 random instances per map/fleet condition on a stated
  32-CPU AMD platform. At 800 agents on their sortation benchmark, their table
  reports roughly 0.012 seconds for a Guided-PIBT planning step versus about
  9.936 seconds per RHCR-ECBS planner invocation. This three-order-of-magnitude
  spread at the same fleet size demonstrates why algorithm and invocation
  semantics must precede any runtime coefficient.
  https://doi.org/10.1609/aaai.v38i18.30054
- Ma et al. use reservation-table path planning and report paths for hundreds of
  agents and thousands of tasks in seconds, again tying runtime claims to an
  explicit algorithm and experimental platform.
  https://doi.org/10.1609/aaai.v33i01.33017651

The thesis should follow this practice by reporting wall-clock planner
response, CPU/resource demand, invocation frequency, hardware, workload, and
solution behavior separately.
