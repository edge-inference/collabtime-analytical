#!/usr/bin/env python3
"""
Analytical Model Runner
Main entry point for running DSM vs centralized performance analysis.
"""

import argparse
import sys
import time
from pathlib import Path
import logging

try:
    from comparison import DSMCentralizedComparison
except ImportError:
    print("Error: Cannot import analytical modules.")
    print("Make sure you're in the analytical/ directory.")
    sys.exit(1)


def main():
    """Main entry point for analytical analysis."""
    parser = argparse.ArgumentParser(
        description='Run analytical performance analysis for DSM vs Centralized warehouse systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis with default config
  python run.py
  
  # Use custom config and output directory
  python run.py -c my_config.yaml -o results/custom/
  
  # Quick analysis (skip plots)
  python run.py --no-plots
  
  # Verbose output with timing
  python run.py -v --timing
  
Output:
  - Text report with performance comparison
  - Plots showing crossover points and advantages
  - Recommendations for architecture choice
        """
    )
    
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--output', '-o', default='results/',
                       help='Output directory (default: results/)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--timing', action='store_true',
                       help='Show timing information')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate only text report, no detailed analysis')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('AnalyticalRunner')
    
    # Validate inputs
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Warehouse DSM vs Centralized Analytical Analysis ===")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    
    start_time = time.time()
    
    try:
        # Initialize comparison engine
        logger.info("Initializing analytical models...")
        comparison = DSMCentralizedComparison(str(config_path))
        
        if args.report_only:
            # Quick analysis
            logger.info("Running quick comparison analysis...")
            result = comparison.comprehensive_comparison()
            
            # Generate report
            report = comparison.generate_summary_report(result)
            print("\n" + report)
            
            # Save report
            with open(output_dir / 'quick_report.txt', 'w') as f:
                f.write(report)
                
        else:
            # Full analysis
            logger.info("Running comprehensive analysis...")
            
            # Individual analyses
            fleet_sizes = comparison.config['system']['fleet_sizes']
            arrival_rates = comparison.config['system']['arrival_rates']
            
            logger.info("1. Analyzing propagation times...")
            prop_analysis = comparison.propagation_time_analysis(fleet_sizes)
            
            logger.info("2. Analyzing total latency...")
            latency_analysis = comparison.total_latency_analysis(fleet_sizes, arrival_rates)
            
            logger.info("3. Analyzing stability boundaries...")
            stability_analysis = comparison.stability_boundary_analysis(fleet_sizes)
            
            logger.info("4. Analyzing Age of Information...")
            gossip_periods = [50, 100, 150, 200, 300, 500]  # milliseconds
            aoi_analysis = comparison.aoi_violation_analysis(gossip_periods)
            
            # Generate comprehensive result
            result = comparison.comprehensive_comparison()
            
            # Generate report
            report = comparison.generate_summary_report(result)
            print("\n" + report)
            
            # Save detailed results
            logger.info("Saving detailed results...")
            
            # Text report
            with open(output_dir / 'comprehensive_report.txt', 'w') as f:
                f.write(report)
                f.write("\n\n=== Detailed Analysis ===\n")
                f.write(f"Propagation Analysis: {prop_analysis}\n\n")
                f.write(f"Stability Analysis: {stability_analysis}\n\n")
                f.write(f"AoI Analysis: {aoi_analysis}\n")
            
            # Generate plots
            if not args.no_plots:
                logger.info("Generating visualization plots...")
                try:
                    from visualize import AnalyticalVisualizer
                    visualizer = AnalyticalVisualizer(str(config_path))
                    if args.output != 'results/':
                        visualizer.output_dir = output_dir / 'plots'
                        visualizer.output_dir.mkdir(exist_ok=True)
                    
                    plot_files = visualizer.generate_all_plots(comparison)
                    
                    if plot_files:
                        logger.info(f"Generated {len(plot_files)} plots:")
                        for plot_file in plot_files:
                            if plot_file:
                                logger.info(f"  {plot_file}")
                    else:
                        logger.warning("No plots generated")
                        
                except ImportError:
                    logger.warning("Plot generation skipped - missing dependencies (numpy, matplotlib)")
                except Exception as e:
                    logger.error(f"Plot generation failed: {e}")
        
        # Timing information
        if args.timing:
            elapsed = time.time() - start_time
            logger.info(f"Analysis completed in {elapsed:.2f} seconds")
        
        # Final recommendations
        print("\n=== Key Findings ===")
        
        bottleneck_threshold = (
            result.performance_advantage.get('scheduler_bottleneck_threshold')
            if 'result' in locals()
            else None
        )
        max_fleet = max(comparison.config['system']['fleet_sizes'])
        
        if bottleneck_threshold and bottleneck_threshold <= max_fleet // 2:
            print("The scheduler becomes capacity-limiting at medium fleet sizes")
            print(
                "First sampled scheduler-bottleneck threshold: "
                f"{bottleneck_threshold} robots"
            )
        elif bottleneck_threshold:
            print(
                "First sampled scheduler-bottleneck threshold: "
                f"{bottleneck_threshold} robots"
            )
            print("Review scheduler sensitivity before drawing a deployment boundary")
        else:
            print("The scheduler is not capacity-limiting in the analyzed range")
        
        if 'result' in locals():
            throughput_adv = result.performance_advantage.get('max_throughput_improvement', 1.0)
            if throughput_adv > 1.2:
                print(f"Conditional throughput gain: {throughput_adv:.2f}x")
            
            latency_adv = result.performance_advantage.get('avg_latency_improvement', 1.0)
            if latency_adv is not None and latency_adv > 1.1:
                print(f"✓ Latency improvements available: {latency_adv:.2f}x average reduction")
        
        print(f"\nResults saved to: {output_dir}")
        
        # Next steps recommendation
        print("\n=== Next Steps ===")
        if bottleneck_threshold and bottleneck_threshold <= max_fleet:
            print("1. Proceed with Mesa+LF simulation for validation")
            print("2. Test DSM with halo gossip parameters from analysis")
            print("3. Calibrate scheduler operation cost and validate the threshold")
        else:
            print("1. Optimize DSM parameters (gossip fanout, tile size)")
            print("2. Consider hybrid centralized-DSM approach")
            print("3. Re-run analysis with adjusted parameters")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
