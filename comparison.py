#!/usr/bin/env python3
"""
DSM vs Centralized Performance Analysis
Compares distributed shared memory against centralized coordination.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass

from models import (
    SystemParams, NetworkParams, QueueParams, PerformanceModel
)


@dataclass 
class ComparisonResult:
    """Results of DSM vs centralized comparison."""
    fleet_sizes: List[int]
    crossover_point: int  # first sampled propagation-delay advantage
    central_metrics: Dict
    dsm_metrics: Dict
    performance_advantage: Dict
    stability_comparison: Dict


class DSMCentralizedComparison:
    """Analyzes performance differences between DSM and centralized architectures."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration.
        """
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_logging()
        self.setup_models()
        
    def load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Extract parameter sets
        sys_cfg = self.config['system']
        net_cfg = self.config['network']
        queue_cfg = self.config['queueing']
        self.aoi_target = self.config.get('aoi', {}).get('target_freshness', 300.0)
        
        self.system_params = SystemParams(
            fleet_size=16,
            arrival_rate=0.001,
            warehouse_width=sys_cfg['warehouse']['width'],
            warehouse_height=sys_cfg['warehouse']['height'],
            robot_speed=sys_cfg['robot']['speed'],
            edge_capacity=sys_cfg['edge_capacity'],
            node_capacity=sys_cfg['node_capacity'],
            t_work=sys_cfg['robot']['work_time'],
            t_traverse=sys_cfg['warehouse']['cell_length'] / sys_cfg['robot']['speed'],
            expected_path_cells=sys_cfg['warehouse'].get('expected_path_cells'),
        )
        
        self.network_params = NetworkParams(
            hop_delay=net_cfg['hop_delay'],
            serialization_delay=net_cfg['serialization_delay'],
            batch_period=net_cfg['central']['batch_period'],
            tree_depth=net_cfg['central']['tree_depth'],
            solver_a=net_cfg['central']['solver_complexity']['a'],
            solver_b=net_cfg['central']['solver_complexity']['b'],
            solver_c=net_cfg['central']['solver_complexity']['c'],
            gossip_fanout=net_cfg['dsm']['gossip_fanout'],
            gossip_period=net_cfg['dsm']['gossip_period'],
            tile_hops=net_cfg['dsm']['tile_hops'],
            claim_rtt=net_cfg['hop_delay'] * 2,
            central_handshake_rtt=net_cfg['central'].get('handshake_rtt', 0.0),
            dsm_handshake_rtt=net_cfg['dsm'].get('handshake_rtt', 0.0),
            conflict_probability=net_cfg['dsm'].get('conflict_probability', 0.0),
            scheduler_replicas=net_cfg['central'].get('scheduler_replicas', 1),
            scheduler_demand_base_ms=net_cfg['central'].get(
                'scheduler_demand_base_ms',
                net_cfg['central'].get('scheduler_service_base_ms', 0.0),
            ),
            scheduler_demand_per_robot_ms=net_cfg['central'].get(
                'scheduler_demand_per_robot_ms',
                net_cfg['central'].get('scheduler_service_per_robot_ms', 0.0),
            ),
        )
        
        self.queue_params = QueueParams(
            arrival_cv_squared=queue_cfg['arrival_cv_squared'],
            service_cv_squared=queue_cfg['service_cv_squared'],
            max_utilization=queue_cfg['max_utilization']
        )
        
    def setup_logging(self):
        """Setup logging for analysis."""   
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DSMComparison')
        
    def setup_models(self):
        """Initialize analytical models."""
        self.performance_model = PerformanceModel(
            self.system_params,
            self.network_params,
            self.queue_params,
            aoi_target=self.aoi_target,
        )
        
    def propagation_time_analysis(self, fleet_sizes: List[int]) -> Dict:
        """Compare propagation times across fleet sizes."""
        self.logger.info("Analyzing propagation times...")
        
        central_times = []
        dsm_times = []
        
        for n in fleet_sizes:
            central_prop = self.performance_model.prop_model.central_propagation_time(n)
            dsm_prop = self.performance_model.prop_model.dsm_propagation_time(n)
            
            central_times.append(central_prop["total"])
            dsm_times.append(dsm_prop["total"])
            
        return {
            "fleet_sizes": fleet_sizes,
            "central_propagation": central_times,
            "dsm_propagation": dsm_times,
            "crossover_point": self._find_crossover(fleet_sizes, central_times, dsm_times)
        }
    
    def total_latency_analysis(self, fleet_sizes: List[int], arrival_rates: List[float]) -> Dict:
        """Compare total latency across operating conditions."""
        self.logger.info("Analyzing total latency...")
        
        central_latency = np.zeros((len(fleet_sizes), len(arrival_rates)))
        dsm_latency = np.zeros((len(fleet_sizes), len(arrival_rates)))
        stability_mask = np.zeros((len(fleet_sizes), len(arrival_rates)), dtype=bool)
        
        for i, n in enumerate(fleet_sizes):
            for j, rate in enumerate(arrival_rates):
                central_prop = self.performance_model.prop_model.central_propagation_time(n)["total"]
                dsm_prop = self.performance_model.prop_model.dsm_propagation_time(n)["total"]
                
                central_coord_limit = self.performance_model.prop_model.central_scheduler_capacity(n)
                central_stable = self.performance_model.stability.max_stable_arrival_rate(
                    n,
                    central_prop,
                    bottleneck_rates=[central_coord_limit],
                )
                dsm_stable = self.performance_model.stability.max_stable_arrival_rate(n, dsm_prop)
                
                if rate < min(central_stable, dsm_stable):
                    central_result = self.performance_model.total_latency(n, rate, "central")
                    dsm_result = self.performance_model.total_latency(n, rate, "dsm")

                    if np.isfinite(central_result["total"]) and np.isfinite(dsm_result["total"]):
                        stability_mask[i, j] = True
                        central_latency[i, j] = central_result["total"]
                        dsm_latency[i, j] = dsm_result["total"]
                    else:
                        central_latency[i, j] = np.inf
                        dsm_latency[i, j] = np.inf
                else:
                    # Mark as unstable
                    central_latency[i, j] = np.inf
                    dsm_latency[i, j] = np.inf
        
        improvement_factor = np.ones_like(central_latency, dtype=float)
        valid = (
            stability_mask
            & np.isfinite(central_latency)
            & np.isfinite(dsm_latency)
            & (dsm_latency > 0.001)
        )
        improvement_factor[valid] = central_latency[valid] / dsm_latency[valid]

        return {
            "fleet_sizes": fleet_sizes,
            "arrival_rates": arrival_rates,
            "central_latency": central_latency,
            "dsm_latency": dsm_latency,
            "stability_mask": stability_mask,
            "improvement_factor": improvement_factor
        }
    
    def stability_boundary_analysis(self, fleet_sizes: List[int]) -> Dict:
        """Compare stability boundaries (max throughput)."""
        self.logger.info("Analyzing stability boundaries...")
        
        central_limits = []
        dsm_limits = []
        
        for n in fleet_sizes:
            central_prop = self.performance_model.prop_model.central_propagation_time(n)["total"]
            dsm_prop = self.performance_model.prop_model.dsm_propagation_time(n)["total"]
            
            central_coord_limit = self.performance_model.prop_model.central_scheduler_capacity(n)
            central_limit = self.performance_model.stability.max_stable_arrival_rate(
                n,
                central_prop,
                bottleneck_rates=[central_coord_limit],
            )
            dsm_limit = self.performance_model.stability.max_stable_arrival_rate(n, dsm_prop)
            
            central_limits.append(central_limit)
            dsm_limits.append(dsm_limit)
        
        return {
            "fleet_sizes": fleet_sizes,
            "central_limits": central_limits,
            "dsm_limits": dsm_limits,
            "throughput_advantage": [d/c if c > 0 else 0 for d, c in zip(dsm_limits, central_limits)]
        }
    
    def aoi_violation_analysis(self, gossip_periods: List[float]) -> Dict:
        """Analyze Age of Information violation probability."""
        self.logger.info("Analyzing AoI violations...")
        
        aoi_model = self.performance_model.aoi_model
        violations = []
        
        for period in gossip_periods:
            transmission_delay = self.network_params.hop_delay * self.network_params.tile_hops
            violation_prob = aoi_model.periodic_violation_probability(
                period,
                transmission_delay,
            )
            violations.append(violation_prob)
        
        return {
            "gossip_periods": gossip_periods,
            "violation_probabilities": violations,
            "target_freshness": aoi_model.target_freshness
        }
    
    def sensitivity_analysis(self, base_fleet_size: int = 16) -> Dict:
        """Perform sensitivity analysis on key parameters."""
        self.logger.info("Performing sensitivity analysis...")
        
        sensitivity_config = self.config.get('sensitivity', {})
        parameters = sensitivity_config.get('parameters', {})
        
        results = {}
        
        for param_name, param_values in parameters.items():
            param_results = {"values": param_values, "central_prop": [], "dsm_prop": []}
            
            for value in param_values:
                # Modify parameter temporarily
                original_value = self._get_parameter(param_name)
                self._set_parameter(param_name, value)
                
                central_prop_dict = self.performance_model.prop_model.central_propagation_time(base_fleet_size)
                dsm_prop_dict = self.performance_model.prop_model.dsm_propagation_time(base_fleet_size)
                
                param_results["central_prop"].append(central_prop_dict["total"])
                param_results["dsm_prop"].append(dsm_prop_dict["total"])
                
                # Restore original value
                self._set_parameter(param_name, original_value)
            
            results[param_name] = param_results
        
        return results
    
    def comprehensive_comparison(self) -> ComparisonResult:
        """Run comprehensive comparison analysis."""
        self.logger.info("Running comprehensive DSM vs centralized comparison...")
        
        fleet_sizes = self.config['system']['fleet_sizes']
        arrival_rates = self.config['system']['arrival_rates']
        
        # Core analyses
        prop_analysis = self.propagation_time_analysis(fleet_sizes)
        latency_analysis = self.total_latency_analysis(fleet_sizes, arrival_rates)
        stability_analysis = self.stability_boundary_analysis(fleet_sizes)
        
        # Find crossover point
        crossover = prop_analysis["crossover_point"]
        
        # Performance advantage summary
        advantage_summary = self._summarize_advantages(
            prop_analysis, latency_analysis, stability_analysis
        )
        
        return ComparisonResult(
            fleet_sizes=fleet_sizes,
            crossover_point=crossover,
            central_metrics={
                "propagation": prop_analysis["central_propagation"],
                "stability": stability_analysis["central_limits"]
            },
            dsm_metrics={
                "propagation": prop_analysis["dsm_propagation"],
                "stability": stability_analysis["dsm_limits"]
            },
            performance_advantage=advantage_summary,
            stability_comparison=stability_analysis
        )
    
    def _find_crossover(self, fleet_sizes: List[int], central_values: List[float], 
                       dsm_values: List[float]) -> int:
        """Find crossover point where DSM becomes better."""
        for i, n in enumerate(fleet_sizes):
            if dsm_values[i] < central_values[i]:
                return n
        return fleet_sizes[-1]  # If no crossover found
    
    def _summarize_advantages(self, prop_analysis: Dict, latency_analysis: Dict, 
                            stability_analysis: Dict) -> Dict:
        """Summarize performance advantages."""
        finite_latency_improvements = latency_analysis["improvement_factor"][
            latency_analysis["stability_mask"]
            & np.isfinite(latency_analysis["improvement_factor"])
        ]
        avg_latency_improvement = (
            float(np.mean(finite_latency_improvements))
            if finite_latency_improvements.size
            else None
        )
        scheduler_bottleneck_threshold = next(
            (
                fleet_size
                for fleet_size, advantage in zip(
                    stability_analysis["fleet_sizes"],
                    stability_analysis["throughput_advantage"],
                )
                if advantage > 1.0 + 1e-12
            ),
            None,
        )

        return {
            "propagation_crossover": prop_analysis["crossover_point"],
            "scheduler_bottleneck_threshold": scheduler_bottleneck_threshold,
            "max_throughput_improvement": max(stability_analysis["throughput_advantage"]),
            "avg_latency_improvement": avg_latency_improvement,
            "latency_sample_count": int(finite_latency_improvements.size),
            "stability_advantage_count": sum(
                1 for adv in stability_analysis["throughput_advantage"] if adv > 1.0
            )
        }
    
    def _get_parameter(self, param_name: str):
        """Get current parameter value."""
        if param_name == "hop_delay":
            return self.network_params.hop_delay
        elif param_name == "solver_complexity_a":
            return self.network_params.solver_a
        elif param_name == "gossip_fanout":
            return self.network_params.gossip_fanout
        elif param_name == "conflict_probability":
            return getattr(self.network_params, 'conflict_probability', 0.0)
        else:
            return None
    
    def _set_parameter(self, param_name: str, value):
        """Set parameter value temporarily."""
        if param_name == "hop_delay":
            self.network_params.hop_delay = value
            # Recreate models with new parameters
            self.performance_model = PerformanceModel(
                self.system_params,
                self.network_params,
                self.queue_params,
                aoi_target=self.aoi_target,
            )
        elif param_name == "solver_complexity_a":
            self.network_params.solver_a = value
            self.performance_model = PerformanceModel(
                self.system_params,
                self.network_params,
                self.queue_params,
                aoi_target=self.aoi_target,
            )
        elif param_name == "gossip_fanout":
            self.network_params.gossip_fanout = value
            # Recreate models with new parameters
            self.performance_model = PerformanceModel(
                self.system_params,
                self.network_params,
                self.queue_params,
                aoi_target=self.aoi_target,
            )
        elif param_name == "conflict_probability":
            self.network_params.conflict_probability = value
            # Recreate models with new parameters
            self.performance_model = PerformanceModel(
                self.system_params,
                self.network_params,
                self.queue_params,
                aoi_target=self.aoi_target,
            )
    
    def generate_summary_report(self, result: ComparisonResult) -> str:
        """Generate text summary of comparison results."""
        report = []
        report.append("=== DSM vs Centralized Performance Analysis ===\n")
        
        propagation_crossover = result.performance_advantage['propagation_crossover']
        scheduler_bottleneck_threshold = result.performance_advantage[
            'scheduler_bottleneck_threshold'
        ]
        report.append(
            f"First sampled propagation-delay advantage: "
            f"{propagation_crossover} robots"
        )
        if scheduler_bottleneck_threshold is None:
            report.append(
                "Scheduler-bottleneck threshold: not reached in the analyzed "
                "fleet range\n"
            )
        else:
            report.append(
                f"First sampled scheduler-bottleneck threshold under configured "
                f"scheduler demand: {scheduler_bottleneck_threshold} robots\n"
            )
        
        report.append("Performance Advantages:")
        report.append(
            f"  First sampled propagation advantage: {propagation_crossover} robots"
        )
        report.append(f"  Max throughput improvement: {result.performance_advantage['max_throughput_improvement']:.2f}x")
        latency_improvement = result.performance_advantage['avg_latency_improvement']
        latency_samples = result.performance_advantage['latency_sample_count']
        if latency_improvement is None:
            report.append("  Average latency improvement: N/A (no jointly stable points)")
        else:
            report.append(
                f"  Average latency improvement: {latency_improvement:.3f}x "
                f"({latency_samples} jointly stable points)"
            )
        report.append(
            "  Fleet sizes with scheduler-limited centralized capacity: "
            f"{result.performance_advantage['stability_advantage_count']}\n"
        )
        
        report.append("Interpretation:")
        if scheduler_bottleneck_threshold is not None:
            report.append(
                f"  The scheduler is capacity-limiting from sampled N="
                f"{scheduler_bottleneck_threshold}"
            )
            report.append(
                "  Treat this threshold as conditional on scheduler demand"
            )
        else:
            report.append(
                "  The scheduler is not capacity-limiting in the analyzed range"
            )
        
        if result.performance_advantage['max_throughput_improvement'] > 1.2:
            report.append("  Significant conditional throughput gains are possible")
        
        return "\n".join(report)


def main():
    """Main entry point for comparison analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DSM vs Centralized Performance Analysis')
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', '-o', default='results/',
                       help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run analysis
    comparison = DSMCentralizedComparison(args.config)
    result = comparison.comprehensive_comparison()
    
    # Generate report
    report = comparison.generate_summary_report(result)
    print(report)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nResults saved to {output_dir}")
    return result


if __name__ == '__main__':
    main()
