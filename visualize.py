#!/usr/bin/env python3
"""
Visualization and Analysis Tools
Generates plots and analysis for DSM vs centralized performance comparison.
"""

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import yaml
    from pathlib import Path
    from typing import Dict, List, Optional
    import logging
    
    from comparison import DSMCentralizedComparison, ComparisonResult
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    print("Install missing packages: pip install numpy matplotlib seaborn pyyaml")


class AnalyticalVisualizer:
    """Creates visualizations for analytical model results."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize visualizer with configuration."""
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_plotting()
        
    def load_config(self):
        """Load plotting configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.plot_config = self.config.get('plots', {})
        except FileNotFoundError:
            # Use defaults if config not found
            self.config = {}
            self.plot_config = {
                'figsize': [12, 8],
                'dpi': 300,
                'style': 'default',
                'colors': {'centralized': '#1f77b4', 'dsm': '#ff7f0e', 'unstable': '#d62728'},
                'output_dir': 'results/analytical',
                'formats': ['png']
            }
    
    def setup_plotting(self):
        """Setup matplotlib and seaborn styling."""
        try:
            # Use clean matplotlib style without background
            plt.style.use(self.plot_config.get('style', 'default'))
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['savefig.facecolor'] = 'white'
            plt.rcParams['savefig.edgecolor'] = 'none'
            plt.rcParams['axes.edgecolor'] = 'black'
            plt.rcParams['axes.linewidth'] = 0.8
            # Paper-friendly font sizes
            base_fs = self.plot_config.get('font_size', 12)
            plt.rcParams.update({
                'font.size': base_fs,
                'axes.titlesize': base_fs + 2,
                'axes.labelsize': base_fs + 1,
                'legend.fontsize': base_fs - 1,
                'xtick.labelsize': base_fs - 1,
                'ytick.labelsize': base_fs - 1,
            })
            # Cache for per-axes application
            self._base_fs = base_fs
            
            # Create output directory
            self.output_dir = Path(self.plot_config.get('output_dir', 'results/analytical'))
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Plot setup failed: {e}")
            self.output_dir = Path('results/analytical')
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _style_axes(self, ax):
        """Apply consistent axis font sizes and spine styling."""
        try:
            fs = getattr(self, '_base_fs', 12)
            ax.xaxis.label.set_size(fs + 2)
            ax.yaxis.label.set_size(fs + 2)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
        except Exception:
            pass
    
    def plot_propagation_vs_fleet_size(self, comparison_result: ComparisonResult, 
                                     show_crossover: bool = True) -> str:
        """Plot propagation time vs fleet size."""
        try:
            fig, ax = plt.subplots(figsize=self.plot_config.get('figsize', [12, 8]))
            
            fleet_sizes = comparison_result.fleet_sizes
            central_prop = comparison_result.central_metrics['propagation']
            dsm_prop = comparison_result.dsm_metrics['propagation']
            
            colors = self.plot_config.get('colors', {})
            
            ax.plot(fleet_sizes, central_prop, 'o-', 
                   color=colors.get('centralized', 'blue'),
                   label='Centralized', linewidth=2, markersize=6)
            ax.plot(fleet_sizes, dsm_prop, 's-', 
                   color=colors.get('dsm', 'orange'),
                   label='DSM', linewidth=2, markersize=6)
            
            # Mark crossover point
            if show_crossover and comparison_result.crossover_point:
                crossover = comparison_result.crossover_point
                if crossover in fleet_sizes:
                    idx = fleet_sizes.index(crossover)
                    ax.axvline(x=crossover, color='red', linestyle='--', alpha=0.7,
                              label=f'Crossover: {crossover}')
                    ax.plot(crossover, dsm_prop[idx], 'ro', markersize=10, alpha=0.7)
            
            ax.set_xlabel('Fleet Size (N)')
            ax.set_ylabel('Propagation Time (ms)')
            leg = ax.legend(loc='center right')
            if leg:
                try:
                    leg.set_title(None)
                    leg._legend_box.sep = 6
                except Exception:
                    pass
            ax.grid(True, alpha=0.3)
            self._style_axes(ax)
            
            # No additional annotation label for crossover
            
            filename = self.output_dir / 'propagation_vs_fleet_size'
            self._save_figure(fig, filename)
            plt.close(fig)
            
            return str(filename) + '.png'
            
        except Exception as e:
            print(f"Error creating propagation plot: {e}")
            return ""
    
    def plot_stability_boundaries(self, comparison_result: ComparisonResult) -> str:
        """Plot maximum stable arrival rates."""
        try:
            fig, ax = plt.subplots(figsize=self.plot_config.get('figsize', [12, 8]))
            
            fleet_sizes = comparison_result.fleet_sizes
            # Convert tasks/ms to tasks/s for plotting
            central_limits = [x * 1000.0 for x in comparison_result.stability_comparison['central_limits']]
            dsm_limits = [x * 1000.0 for x in comparison_result.stability_comparison['dsm_limits']]
            
            colors = self.plot_config.get('colors', {})
            
            ax.plot(fleet_sizes, central_limits, 'o-', 
                   color=colors.get('centralized', 'blue'),
                   label='Centralized', linewidth=2, markersize=6)
            ax.plot(fleet_sizes, dsm_limits, 's-', 
                   color=colors.get('dsm', 'orange'),
                   label='DSM', linewidth=2, markersize=6)
            
            # Fill area between curves to show advantage
            ax.fill_between(fleet_sizes, central_limits, dsm_limits,
                          where=np.array(dsm_limits) > np.array(central_limits),
                          alpha=0.3, color='green', label='DSM Advantage')
            
            ax.set_xlabel('Fleet Size (N)')
            ax.set_ylabel('$\\lambda_{max}$ (tasks/s)')
            leg = ax.legend()
            if leg:
                try:
                    leg.set_title(None)
                    leg._legend_box.sep = 6
                except Exception:
                    pass
            ax.grid(True, alpha=0.3)
            self._style_axes(ax)
            
            # Add improvement annotation (concise)
            max_improvement = max(comparison_result.stability_comparison['throughput_advantage'])
            # Use larger font for annotation to match global settings
            ann_fs = getattr(self, '_base_fs', 12) + 2
            ax.text(0.05, 0.90, f'Improvement: {max_improvement:.2f}x',
                   transform=ax.transAxes, fontsize=ann_fs,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            filename = self.output_dir / 'stability_boundaries'
            self._save_figure(fig, filename)
            plt.close(fig)
            
            return str(filename) + '.png'
            
        except Exception as e:
            print(f"Error creating stability plot: {e}")
            return ""
    
    def plot_latency_heatmap(self, comparison: DSMCentralizedComparison) -> str:
        """Plot total latency heatmap."""
        try:
            fleet_sizes = self.config['system']['fleet_sizes']
            arrival_rates = self.config['system']['arrival_rates']
            
            # Get latency analysis
            latency_analysis = comparison.total_latency_analysis(fleet_sizes, arrival_rates)
            
            # Create improvement factor heatmap
            improvement = latency_analysis['improvement_factor']
            stability_mask = latency_analysis['stability_mask']
            
            # Mask unstable regions
            masked_improvement = np.where(stability_mask, improvement, np.nan)
            
            # Portrait multi-panel (2x1)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.plot_config.get('figsize', [12, 8]))
            fig.set_size_inches(self.plot_config.get('figsize', [12, 8])[0], self.plot_config.get('figsize', [12, 8])[1] * 2)
            
            # DSM vs Central improvement
            im1 = ax1.imshow(masked_improvement, aspect='auto', origin='lower',
                           cmap='RdYlGn', vmin=0.5, vmax=2.0)
            ax1.set_xlabel('Arrival Rate Index')
            ax1.set_ylabel('Fleet Size Index')
            self._style_axes(ax1)
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Improvement Factor')
            try:
                cbar1.ax.yaxis.label.set_size(self._base_fs)
                cbar1.ax.tick_params(labelsize=self._base_fs - 1)
            except Exception:
                pass
            
            # Stability region
            im2 = ax2.imshow(stability_mask.astype(int), aspect='auto', origin='lower',
                           cmap='RdYlGn_r', alpha=0.7)
            ax2.set_xlabel('Arrival Rate Index')
            ax2.set_ylabel('Fleet Size Index')
            self._style_axes(ax2)
            
            # Add rate/size labels
            n_ticks = 5
            fleet_indices = np.linspace(0, len(fleet_sizes)-1, n_ticks, dtype=int)
            rate_indices = np.linspace(0, len(arrival_rates)-1, n_ticks, dtype=int)
            
            for ax in [ax1, ax2]:
                ax.set_xticks(rate_indices)
                # Show arrival rates in tasks/second
                ax.set_xticklabels([f'{arrival_rates[i]*1000.0:.2f}' for i in rate_indices])
                ax.set_yticks(fleet_indices)
                ax.set_yticklabels([f'{fleet_sizes[i]}' for i in fleet_indices])
            
            plt.tight_layout()
            
            filename = self.output_dir / 'latency_heatmap'
            self._save_figure(fig, filename)
            plt.close(fig)
            
            return str(filename) + '.png'
            
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            return ""
    
    def plot_aoi_violations(self, comparison: DSMCentralizedComparison) -> str:
        """Plot Age of Information violation probability."""
        try:
            gossip_periods = np.linspace(50, 500, 20)  # 50ms to 500ms
            aoi_analysis = comparison.aoi_violation_analysis(gossip_periods)
            
            fig, ax = plt.subplots(figsize=self.plot_config.get('figsize', [12, 8]))
            
            ax.plot(gossip_periods, aoi_analysis['violation_probabilities'], 
                   'o-', linewidth=2, markersize=6, color='red')
            
            # Mark target violation rate
            target_rate = 0.05  # 5%
            ax.axhline(y=target_rate, color='green', linestyle='--', 
                      label=f'Target violation rate: {target_rate*100}%')
            
            # Mark freshness target
            target_fresh = aoi_analysis['target_freshness']
            ax.axvline(x=target_fresh/2, color='blue', linestyle='--', alpha=0.7,
                      label=f'Optimal period: {target_fresh/2}ms')
            
            ax.set_xlabel('Gossip Period (ms)')
            ax.set_ylabel('AoI Violation Probability')
            leg = ax.legend()
            if leg:
                try:
                    leg.set_title(None)
                    leg._legend_box.sep = 6
                except Exception:
                    pass
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            self._style_axes(ax)
            
            filename = self.output_dir / 'aoi_violations'
            self._save_figure(fig, filename)
            plt.close(fig)
            
            return str(filename) + '.png'
            
        except Exception as e:
            print(f"Error creating AoI plot: {e}")
            return ""
    
    def plot_sensitivity_analysis(self, comparison: DSMCentralizedComparison) -> str:
        """Plot parameter sensitivity analysis."""
        try:
            sensitivity = comparison.sensitivity_analysis()
            
            n_params = len(sensitivity)
            fig, axes = plt.subplots(2, 2, figsize=self.plot_config.get('figsize', [12, 8]))
            fig.set_size_inches(self.plot_config.get('figsize', [12, 8])[0] * 2, self.plot_config.get('figsize', [12, 8])[1] * 1.5)
            axes = axes.flatten()
            
            colors = self.plot_config.get('colors', {})
            
            for i, (param_name, data) in enumerate(sensitivity.items()):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                values = data['values']
                central_prop = data['central_prop']
                dsm_prop = data['dsm_prop']
                
                ax.plot(values, central_prop, 'o-', 
                       color=colors.get('centralized', 'blue'),
                       label='Centralized', linewidth=2)
                ax.plot(values, dsm_prop, 's-', 
                       color=colors.get('dsm', 'orange'),
                       label='DSM', linewidth=2)
                
                ax.set_xlabel(param_name.replace('_', ' ').title())
                ax.set_ylabel('Propagation Time (ms)')
                # No title for paper style
                leg = ax.legend()
                if leg:
                    try:
                        leg.set_title(None)
                        leg._legend_box.sep = 6
                    except Exception:
                        pass
                ax.grid(True, alpha=0.3)
                self._style_axes(ax)
            
            # Hide unused subplots
            for i in range(len(sensitivity), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            filename = self.output_dir / 'sensitivity_analysis'
            self._save_figure(fig, filename)
            plt.close(fig)
            
            return str(filename) + '.png'
            
        except Exception as e:
            print(f"Error creating sensitivity plot: {e}")
            return ""

    def plot_sensitivity_gossip_fanout(self, comparison: DSMCentralizedComparison) -> str:
        """Plot propagation vs gossip fanout (single panel)."""
        try:
            sensitivity = comparison.sensitivity_analysis()
            data = sensitivity.get('gossip_fanout')
            if not data:
                return ""
            values = data['values']
            central_prop = data['central_prop']
            dsm_prop = data['dsm_prop']

            fig, ax = plt.subplots(figsize=self.plot_config.get('figsize', [12, 8]))

            colors = self.plot_config.get('colors', {})
            ax.plot(values, central_prop, 'o-', color=colors.get('centralized', 'blue'),
                    label='Centralized', linewidth=2, markersize=6)
            ax.plot(values, dsm_prop, 's-', color=colors.get('dsm', 'orange'),
                    label='DSM', linewidth=2, markersize=6)

            ax.set_xlabel('Gossip Fanout (f)')
            ax.set_ylabel('Propagation Time (ms)')
            leg = ax.legend()
            if leg:
                try:
                    leg.set_title(None)
                    leg._legend_box.sep = 6
                except Exception:
                    pass
            ax.grid(True, alpha=0.3)
            self._style_axes(ax)

            filename = self.output_dir / 'sensitivity_gossip_fanout'
            self._save_figure(fig, filename)
            plt.close(fig)
            return str(filename) + '.png'
        except Exception as e:
            print(f"Error creating gossip fanout sensitivity plot: {e}")
            return ""
    
    def plot_crossover_analysis(self, comparison_result: ComparisonResult) -> str:
        """Plot detailed crossover analysis."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.plot_config.get('figsize', [12, 8]))
            fig.set_size_inches(self.plot_config.get('figsize', [12, 8])[0], self.plot_config.get('figsize', [12, 8])[1] * 2)
            
            fleet_sizes = comparison_result.fleet_sizes
            central_prop = comparison_result.central_metrics['propagation']
            dsm_prop = comparison_result.dsm_metrics['propagation']
            central_stab = comparison_result.central_metrics['stability']
            dsm_stab = comparison_result.dsm_metrics['stability']
            
            colors = self.plot_config.get('colors', {})
            
            # Propagation time comparison
            ax1.plot(fleet_sizes, central_prop, 'o-', 
                    color=colors.get('centralized', 'blue'),
                    label='Centralized', linewidth=2, markersize=6)
            ax1.plot(fleet_sizes, dsm_prop, 's-', 
                    color=colors.get('dsm', 'orange'),
                    label='DSM', linewidth=2, markersize=6)
            
            crossover = comparison_result.crossover_point
            if crossover and crossover in fleet_sizes:
                ax1.axvline(x=crossover, color='red', linestyle='--', alpha=0.7)
                ax1.text(crossover, max(max(central_prop), max(dsm_prop)) * 0.8,
                        f'Crossover\n{crossover} robots', ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            ax1.set_ylabel('Propagation Time (ms)')
            leg1 = ax1.legend()
            if leg1:
                try:
                    leg1.set_title(None)
                    leg1._legend_box.sep = 6
                except Exception:
                    pass
            ax1.grid(True, alpha=0.3)
            self._style_axes(ax1)
            
            # Stability comparison
            ax2.plot(fleet_sizes, central_stab, 'o-', 
                    color=colors.get('centralized', 'blue'),
                    label='Centralized', linewidth=2, markersize=6)
            ax2.plot(fleet_sizes, dsm_stab, 's-', 
                    color=colors.get('dsm', 'orange'),
                    label='DSM', linewidth=2, markersize=6)
            
            # Convert to tasks/s
            central_stab = [x * 1000.0 for x in central_stab]
            dsm_stab = [x * 1000.0 for x in dsm_stab]
            ax2.set_xlabel('Fleet Size (N)')
            ax2.set_ylabel('$\\lambda_{max}$ (tasks/s)')
            leg2 = ax2.legend()
            if leg2:
                try:
                    leg2.set_title(None)
                    leg2._legend_box.sep = 6
                except Exception:
                    pass
            ax2.grid(True, alpha=0.3)
            self._style_axes(ax2)
            
            plt.tight_layout()
            
            filename = self.output_dir / 'crossover_analysis'
            self._save_figure(fig, filename)
            plt.close(fig)
            
            return str(filename) + '.png'
            
        except Exception as e:
            print(f"Error creating crossover plot: {e}")
            return ""
    
    def _save_figure(self, fig, filename: Path):
        """Save figure in configured formats."""
        formats = self.plot_config.get('formats', ['png'])
        dpi = self.plot_config.get('dpi', 300)
        
        for fmt in formats:
            fig.savefig(f"{filename}.{fmt}", dpi=dpi, bbox_inches='tight')
    
    def generate_all_plots(self, comparison: DSMCentralizedComparison) -> List[str]:
        """Generate all analysis plots."""
        print("Generating analytical plots...")
        
        # Run comprehensive comparison
        result = comparison.comprehensive_comparison()
        
        plot_files = []
        
        # Generate each plot type
        plots_to_generate = self.plot_config.get('generate', [
            'propagation_vs_fleet_size', 'stability_boundaries', 
            'latency_heatmap', 'aoi_violations'
        ])
        
        if 'propagation_vs_fleet_size' in plots_to_generate:
            plot_files.append(self.plot_propagation_vs_fleet_size(result))
        
        if 'stability_boundaries' in plots_to_generate:
            plot_files.append(self.plot_stability_boundaries(result))
        
        if 'latency_heatmap' in plots_to_generate:
            plot_files.append(self.plot_latency_heatmap(comparison))
        
        if 'aoi_violations' in plots_to_generate:
            plot_files.append(self.plot_aoi_violations(comparison))
        
        # Crossover analysis removed from default set; enable via config if needed
        
        # Optional: single sensitivity plot (gossip fanout)
        if 'sensitivity_gossip_fanout' in plots_to_generate:
            try:
                plot_files.append(self.plot_sensitivity_gossip_fanout(comparison))
            except Exception as e:
                print(f"Skipping sensitivity_gossip_fanout: {e}")
        
        # Filter out empty filenames
        plot_files = [f for f in plot_files if f]
        
        print(f"Generated {len(plot_files)} plots in {self.output_dir}")
        return plot_files


def main():
    """Main entry point for visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate analytical model visualizations')
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory override')
    
    args = parser.parse_args()
    
    try:
        # Create comparison and visualizer
        comparison = DSMCentralizedComparison(args.config)
        visualizer = AnalyticalVisualizer(args.config)
        
        if args.output:
            visualizer.output_dir = Path(args.output)
            visualizer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        plot_files = visualizer.generate_all_plots(comparison)
        
        print("\nGenerated plots:")
        for plot_file in plot_files:
            if plot_file:
                print(f"  {plot_file}")
                
    except ImportError:
        print("Error: Required packages not installed.")
        print("Install with: pip install numpy matplotlib seaborn pyyaml")
    except Exception as e:
        print(f"Error generating plots: {e}")


if __name__ == '__main__':
    main()