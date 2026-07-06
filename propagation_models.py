#!/usr/bin/env python3
"""
Propagation Models for Centralized vs DSM Communication
Models information propagation delays in different coordination architectures.
"""

import math
from typing import Dict
from dataclasses import dataclass


@dataclass
class NetworkParams:
    """Network and communication parameters."""
    hop_delay: float  # milliseconds
    serialization_delay: float  # milliseconds
    batch_period: float  # milliseconds (central)
    tree_depth: int  # broadcast tree depth
    solver_a: float  # T_solve(N) quadratic coefficient
    solver_b: float  # T_solve(N) linear coefficient
    solver_c: float  # T_solve(N) constant term
    gossip_fanout: int  # DSM fanout
    gossip_period: float  # milliseconds
    tile_hops: float  # average hops between tiles
    claim_rtt: float  # milliseconds - RTT for task claim confirmation
    central_handshake_rtt: float  # milliseconds - optional central propagation ACK
    dsm_handshake_rtt: float  # milliseconds - optional DSM propagation ACK
    conflict_probability: float  # probability a claim attempt conflicts (0..1)
    scheduler_replicas: int = 1  # centralized scheduler service replicas
    scheduler_service_base_ms: float = 0.0  # base worker-time demand per order
    scheduler_service_per_robot_ms: float = 0.0  # per-robot demand growth per order


class PropagationModel:
    """Models information propagation in centralized vs DSM systems."""
    
    def __init__(self, network_params: NetworkParams):
        self.params = network_params

    def _solver_time(self, fleet_size: int) -> float:
        """Solver time complexity T_solve(N) = a*N^2 + b*N + c."""
        return (
            self.params.solver_a * fleet_size**2
            + self.params.solver_b * fleet_size
            + self.params.solver_c
        )
    
    def claim_time_dsm(self, num_contenders: int = 2, window_size: float = 100.0) -> float:
        """Expected claim time with continuous random backoff and conflict retries.
        
        Args:
            num_contenders: Number of robots contending for the task (k)
            window_size: Size of uniform backoff window in ms (W)
            
        Returns:
            Expected claim time in milliseconds
        """
        # Continuous random backoff per-attempt time: E[T_attempt] = W/(k+1) + RTT
        expected_min_time = window_size / (num_contenders + 1)
        per_attempt = expected_min_time + self.params.claim_rtt

        # Expected attempts under geometric retries with conflict probability p
        p = max(0.0, min(self.params.conflict_probability, 0.99))
        attempts_multiplier = 1.0 / (1.0 - p)  # E[attempts]
        return per_attempt * attempts_multiplier
    
    def claim_time_central(self) -> float:
        """Claim time for centralized (negligible vs solver time)."""
        return 0.0

    def central_scheduler_service_time(self, fleet_size: int) -> float:
        """Aggregate scheduler worker-time demand per order in milliseconds."""
        return (
            self.params.scheduler_service_base_ms
            + self.params.scheduler_service_per_robot_ms * fleet_size
        )

    def central_scheduler_capacity(self, fleet_size: int) -> float:
        """Maximum centralized scheduler throughput in tasks/ms."""
        service_time = self.central_scheduler_service_time(fleet_size)
        if service_time <= 0:
            return float("inf")
        replicas = max(self.params.scheduler_replicas, 1)
        return replicas / service_time

    def central_propagation_time(self, fleet_size: int) -> Dict[str, float]:
        """Propagation time breakdown for centralized system.
        
        Returns:
            Dictionary with time breakdown in milliseconds
        """
        solver_time = self._solver_time(fleet_size)
        broadcast_time = self.params.tree_depth * self.params.hop_delay
        serialization_time = self.params.tree_depth * self.params.serialization_delay
        batch_wait = 0.5 * self.params.batch_period
        handshake_time = self.params.central_handshake_rtt
        total_time = (
            batch_wait
            + solver_time
            + broadcast_time
            + serialization_time
            + handshake_time
        )
        
        return {
            "batch": batch_wait,
            "solver": solver_time,
            "broadcast": broadcast_time,
            "serialization": serialization_time,
            "handshake": handshake_time,
            "total": total_time
        }
    
    def dsm_propagation_time(self, fleet_size: int) -> Dict[str, float]:
        """Propagation time breakdown for DSM system.
        
        Returns:
            Dictionary with time breakdown in milliseconds
        """
        pre_delay = 0.5 * self.params.gossip_period

        tile_hop_time = self.params.tile_hops * self.params.hop_delay
        tile_serialization_time = self.params.tile_hops * self.params.serialization_delay
        tile_time = tile_hop_time
        handshake_time = self.params.dsm_handshake_rtt
        
        # Edge case: single robot needs no gossip
        if fleet_size <= 1:
            return {
                "pre": pre_delay,
                "tile": tile_time,
                "gossip": 0.0,
                "serialization": tile_serialization_time,
                "handshake": handshake_time,
                "total": (
                    pre_delay
                    + tile_time
                    + tile_serialization_time
                    + handshake_time
                ),
            }
        
        # Gossip rounds needed to reach all nodes
        if self.params.gossip_fanout > 1:
            gossip_rounds = math.ceil(math.log(fleet_size) / math.log(self.params.gossip_fanout))
        else:
            gossip_rounds = fleet_size - 1  # Linear propagation (exclude self)
            
        gossip_hop_time = gossip_rounds * self.params.hop_delay
        gossip_serialization_time = gossip_rounds * self.params.serialization_delay
        serialization_time = tile_serialization_time + gossip_serialization_time
        gossip_time = gossip_hop_time
        total_time = (
            pre_delay
            + tile_time
            + gossip_time
            + serialization_time
            + handshake_time
        )
        
        return {
            "pre": pre_delay,
            "tile": tile_time,
            "gossip": gossip_time,
            "serialization": serialization_time,
            "handshake": handshake_time,
            "total": total_time
        }
