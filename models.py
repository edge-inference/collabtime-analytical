#!/usr/bin/env python3
"""
Analytical Model Core Components
Implements queueing theory, network delays, and service time models for warehouse systems.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
from queue_models import QueueModel
from propagation_models import PropagationModel, NetworkParams


@dataclass
class SystemParams:
    """System parameters for analytical model - all timing in milliseconds."""
    fleet_size: int
    arrival_rate: float  # tasks/ms
    warehouse_width: int
    warehouse_height: int
    robot_speed: float  # m/ms
    edge_capacity: int
    node_capacity: int
    t_work: float  # work time at nodes (ms)
    t_traverse: float  # edge traversal time (ms)


@dataclass
class QueueParams:
    """Queueing theory parameters."""
    arrival_cv_squared: float
    service_cv_squared: float
    max_utilization: float

@dataclass
class RoutingParams:
    pickup_fraction: float = 1.0
    delivery_fraction: float = 1.0
    sortation_fraction: float = 0.0
    charging_fraction: float = 0.0


class ArrivalModel:
    """Models task arrivals and service requirements."""
    
    def __init__(self, params: SystemParams):
        self.params = params
        
    def poisson_arrivals(self, lambda_rate: float, duration: float) -> np.ndarray:
        """Generate Poisson arrival times."""
        n_arrivals = np.random.poisson(lambda_rate * duration)
        return np.sort(np.random.uniform(0, duration, n_arrivals))
    
    def service_time_distribution(self, path_length: float, queue_delay: float = 0) -> float:
        """Total service time for a task in milliseconds.
        Assumes path_length is measured in cells; converts via per-cell traverse time."""
        travel_time_ms = path_length * self.params.t_traverse
        return travel_time_ms + self.params.t_work + queue_delay
    
    def expected_path_length(self) -> float:
        """Expected shortest path length in cells (dimensionless).
        Approximate as Manhattan distance fraction of grid size."""
        avg_cells = (self.params.warehouse_width + self.params.warehouse_height) / 3
        return avg_cells
    
    def fleet_utilization(self, arrival_rate: float) -> float:
        """Fleet utilization ρ = λ × E[S] / N (arrival_rate in tasks/ms, service_time in ms)"""
        avg_service_time_ms = self.service_time_distribution(self.expected_path_length())
        return arrival_rate * avg_service_time_ms / self.params.fleet_size


class QueueingModel:
    """Queueing model using G/G/1 and G/G/c approximations."""
    
    def __init__(self, queue_params: QueueParams, system_params: SystemParams):
        self.params = queue_params
        self.system_params = system_params
    
    def edge_time(self, traffic_rate: float, lanes: int = 1) -> float:
        """Total edge time = t_traverse + wait_time (ms)."""
        edge_queue = QueueModel(
            arrival_rate=traffic_rate,  # tasks/ms
            service_time=self.system_params.t_traverse,  # ms
            servers=lanes,
            Ca2=self.params.arrival_cv_squared,
            Cs2=self.params.service_cv_squared
        )
        wait_time_ms = edge_queue.waiting_time()  # ms
        return self.system_params.t_traverse + wait_time_ms
    
    def node_time(self, traffic_rate: float, bays: int = 1) -> float:
        """Total node time = t_work + wait_time (in milliseconds)."""
        node_queue = QueueModel(
            arrival_rate=traffic_rate,  # tasks/ms
            service_time=self.system_params.t_work,  # ms
            servers=bays,
            Ca2=self.params.arrival_cv_squared,
            Cs2=self.params.service_cv_squared
        )
        wait_time_ms = node_queue.waiting_time()  # ms
        return self.system_params.t_work + wait_time_ms
    
    def edge_utilization(self, traffic_rate: float, lanes: int = 1) -> float:
        """ρ_edge = λ/(lanes × μ_edge), clamped to [0,1]"""
        service_rate = lanes / self.system_params.t_traverse  # tasks/ms
        return min(traffic_rate / service_rate, 1.0) if service_rate > 0 else 0.0
    
    def node_utilization(self, traffic_rate: float, bays: int = 1) -> float:
        """ρ_node = λ/(bays × μ_node), clamped to [0,1]"""
        service_rate = bays / self.system_params.t_work  # tasks/ms
        return min(traffic_rate / service_rate, 1.0) if service_rate > 0 else 0.0
    
    def total_completion_time(self, path_edges: List[tuple], node_workload: List[tuple]) -> float:
        """T_completion = Σ(edge_time) + Σ(node_time) - full task completion time (ms)
        
        Args:
            path_edges: List of (edge_id, traffic_rate, traverse_time)
            node_workload: List of (node_id, traffic_rate, work_time, bays)
        """
        total = 0.0
        for _, traffic_rate, _ in path_edges:
            total += self.edge_time(traffic_rate, self.system_params.edge_capacity)
        for _, traffic_rate, _, bays in node_workload:
            total += self.node_time(traffic_rate, bays)
        return total


class AgeOfInformationModel:
    """Models Age of Information (AoI) for data freshness.
    
    AoI measures the age of the most recent information at a destination.
    """
    
    def __init__(self, target_freshness: float = 300.0):
        self.target_freshness = target_freshness  # milliseconds
    
    def periodic_aoi_with_delay(self, update_period: float, transmission_delay: float) -> float:
        """Average AoI for periodic updates with transmission delay.
        
        Args:
            update_period: Time between consecutive updates (Δ) in ms
            transmission_delay: Fixed transmission delay (τ) in ms
            
        Returns:
            Average AoI: E[A] = Δ/2 + τ
        """
        return update_period / 2 + transmission_delay
    

    def violation_probability(self, aoi_mean: float, aoi_std: float = None, 
                          distribution: str = "exponential") -> float:
        """Probability that AoI exceeds target freshness.
        
        Args:
            aoi_mean: Mean Age of Information in ms
            aoi_std: Standard deviation of AoI (optional)
            distribution: exponential
            
        Returns:
            Probability P(A > τ) where τ is target_freshness
        """
        if distribution == "exponential" or aoi_std is None:
            # Exponential distribution (memoryless) P(A > τ) = exp(-τ/mean)
            return np.exp(-self.target_freshness / aoi_mean)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
    
    def required_update_rate(self, target_violation_prob: float = 0.05, 
                             service_time: float = 0) -> float:
        """Required update rate to achieve target violation probability.
        
        Args:
            target_violation_prob: Target probability P(A > τ) to achieve
            service_time: Mean service/transmission time in ms
            
        Returns:
            Required update rate λ in updates/ms
        """
        # For exponential AoI: P(A > τ) = exp(-τ/mean)
        # Solve for mean: mean = -τ/ln(P)
        required_mean = -self.target_freshness / np.log(target_violation_prob)
        
        if service_time <= 0:
            # No service time, just update interval
            # Mean AoI = 2/λ for Poisson process
            # So λ = 2/mean (in updates/ms)
            return 2 / required_mean
        else:
            # Mean AoI = 1/λ + service_time
            # So λ = 1/(mean - service_time)
            adjusted_mean = max(required_mean - service_time, 0.001)
            return 1 / adjusted_mean


class StabilityAnalysis:
    """Analyzes system stability and capacity limits."""
    
    def __init__(self, arrival_model: ArrivalModel, queue_model: QueueingModel):
        self.arrival_model = arrival_model
        self.queue_model = queue_model
    
    def fleet_capacity(self, fleet_size: int, propagation_delay: float = 0) -> float:
        """Effective fleet capacity accounting for coordination overhead.
        
        Args:
            fleet_size: Number of robots in the fleet (N)
            propagation_delay: Coordination delay in milliseconds
            
        Returns:
            Maximum task throughput capacity in tasks/ms
        """
        # Base capacity C_base = N / E[S] (all in ms)
        mean_service_time_ms = self.arrival_model.service_time_distribution(
            self.arrival_model.expected_path_length()
        )
        base_capacity = fleet_size / mean_service_time_ms  # tasks/ms
        
        # Reduce capacity due to coordination delays
        # C_eff = C_base / (1 + T_prop_ms * C_base)
        coordination_factor = 1 + (propagation_delay * base_capacity)
        effective_capacity = base_capacity / coordination_factor
        
        return effective_capacity
    
    def max_stable_arrival_rate(self, fleet_size: int, propagation_delay: float = 0,
                               bottleneck_rates: List[float] = None) -> float:
        """Maximum stable arrival rate considering both global and bottleneck constraints.
        
        Args:
            fleet_size: Number of robots in the fleet
            propagation_delay: Coordination delay in milliseconds
            bottleneck_rates: List of service rates for bottleneck resources (tasks/ms)
            
        Returns:
            Maximum stable arrival rate in tasks/ms
        """
        # Global stability constraint based on fleet capacity
        global_capacity = self.fleet_capacity(fleet_size, propagation_delay)
        global_limit = global_capacity * self.queue_model.params.max_utilization
        
        # Bottleneck stability constraints (if any provided)
        bottleneck_limit = float('inf')
        if bottleneck_rates and len(bottleneck_rates) > 0:
            # Find the minimum service rate among all bottleneck resources
            min_bottleneck_rate = min(bottleneck_rates)
            # Apply utilization constraint to bottleneck
            bottleneck_limit = min_bottleneck_rate * self.queue_model.params.max_utilization
        
        # Take the minimum of global and bottleneck constraints
        return min(global_limit, bottleneck_limit)
    
    def stability_boundary(self, fleet_sizes: List[int], 
                          central_prop_func, dsm_prop_func) -> Tuple[List[float], List[float]]:
        """Compute stability boundaries for central vs DSM."""
        central_limits = []
        dsm_limits = []
        
        for n in fleet_sizes:
            central_prop = central_prop_func(n)["total"] if isinstance(central_prop_func(n), dict) else central_prop_func(n)
            dsm_prop = dsm_prop_func(n)["total"] if isinstance(dsm_prop_func(n), dict) else dsm_prop_func(n)
            
            central_limit = self.max_stable_arrival_rate(n, central_prop)
            dsm_limit = self.max_stable_arrival_rate(n, dsm_prop)
            
            central_limits.append(central_limit)
            dsm_limits.append(dsm_limit)
        
        return central_limits, dsm_limits
    
    def crossover_point(self, fleet_sizes: List[int], 
                       central_metrics: List[float], 
                       dsm_metrics: List[float]) -> Optional[int]:
        """Find crossover point where DSM becomes better than centralized."""
        for i, n in enumerate(fleet_sizes):
            if dsm_metrics[i] < central_metrics[i]:
                return n
        return None


class PerformanceModel:
    """Integrates all models for end-to-end performance analysis."""
    
    def __init__(self, system_params: SystemParams, network_params: NetworkParams, 
                 queue_params: QueueParams, routing_params: RoutingParams = None):
        self.system_params = system_params
        self.network_params = network_params
        self.queue_params = queue_params
        self.routing_params = routing_params or RoutingParams()
        
        self.arrival_model = ArrivalModel(system_params)
        self.queue_model = QueueingModel(queue_params, system_params)
        self.prop_model = PropagationModel(network_params)
        self.aoi_model = AgeOfInformationModel()
        self.stability = StabilityAnalysis(self.arrival_model, self.queue_model)
    
    def total_latency(self, fleet_size: int, arrival_rate: float, 
                     architecture: str = "central") -> Dict[str, float]:
        """Compute total task latency breakdown.
        
        Args:
            fleet_size: Number of robots in the fleet
            arrival_rate: System task arrival rate (tasks/ms)
            architecture: "central" or "dsm" coordination architecture
            
        Returns:
            Dictionary with latency breakdown components in milliseconds
        """
        # Propagation time (coordination overhead)
        if architecture == "central":
            prop_breakdown = self.prop_model.central_propagation_time(fleet_size)
            t_prop = prop_breakdown["total"]
            t_claim = self.prop_model.claim_time_central()
        else:  # DSM
            prop_breakdown = self.prop_model.dsm_propagation_time(fleet_size)
            t_prop = prop_breakdown["total"]
            t_claim = self.prop_model.claim_time_dsm()
        path_length = int(self.arrival_model.expected_path_length())
        edge_traffic = arrival_rate / fleet_size
        traverse_time = 1.0 / self.system_params.robot_speed

        path_edges = [(i, edge_traffic, traverse_time) for i in range(path_length)]

        pickup_node_traffic = arrival_rate * self.routing_params.pickup_fraction
        delivery_node_traffic = arrival_rate * self.routing_params.delivery_fraction
        sortation_node_traffic = arrival_rate * self.routing_params.sortation_fraction
        charging_node_traffic = arrival_rate * self.routing_params.charging_fraction

        node_workload = [
            (0, pickup_node_traffic, self.system_params.t_work, self.system_params.node_capacity),
            (1, delivery_node_traffic, self.system_params.t_work, self.system_params.node_capacity),
        ]
        
        if self.routing_params.sortation_fraction > 0:
            node_workload.append((2, sortation_node_traffic, self.system_params.t_work * 0.8, self.system_params.node_capacity))
        if self.routing_params.charging_fraction > 0:
            node_workload.append((3, charging_node_traffic, self.system_params.t_work * 2.0, self.system_params.node_capacity))

        task_time_ms = self.queue_model.total_completion_time(path_edges, node_workload)        
        total_latency_ms = t_prop + t_claim + task_time_ms
        
        return {
            "propagation": t_prop,
            "claim": t_claim,
            "task": task_time_ms,
            "total": total_latency_ms
        }
    
    def performance_sweep(self, fleet_sizes: List[int], arrival_rates: List[float]) -> Dict:
        """Sweep performance across fleet sizes and arrival rates."""
        results = {
            "fleet_sizes": fleet_sizes,
            "arrival_rates": arrival_rates,
            "central": {"propagation": [], "total": [], "stable": []},
            "dsm": {"propagation": [], "total": [], "stable": []}
        }
        
        for n in fleet_sizes:
            central_prop = self.prop_model.central_propagation_time(n)["total"]
            dsm_prop = self.prop_model.dsm_propagation_time(n)["total"]

            central_limit = self.stability.max_stable_arrival_rate(n, central_prop)
            dsm_limit = self.stability.max_stable_arrival_rate(n, dsm_prop)

            results["central"]["propagation"].append(central_prop)
            results["dsm"]["propagation"].append(dsm_prop)
            results["central"]["stable_limit"].append(central_limit)
            results["dsm"]["stable_limit"].append(dsm_limit)

            central_totals, central_stable = [], []
            dsm_totals, dsm_stable = [], []

            for r in arrival_rates:
                # Central
                if r <= central_limit and r >= 0:
                    central_totals.append(self.total_latency(n, r, "central")["total"])
                    central_stable.append(True)
                else:
                    central_totals.append(None)
                    central_stable.append(False)

                # DSM
                if r <= dsm_limit and r >= 0:
                    dsm_totals.append(self.total_latency(n, r, "dsm")["total"])
                    dsm_stable.append(True)
                else:
                    dsm_totals.append(None)
                    dsm_stable.append(False)

            results["central"]["total"].append(central_totals)
            results["central"]["stable"].append(central_stable)
            results["dsm"]["total"].append(dsm_totals)
            results["dsm"]["stable"].append(dsm_stable)

        return results
