"""evaluation package — metrics, statistics, visualization, and tables."""

__all__ = [
    # Statistics & hypothesis tests
    'benjamini_hochberg', 'cliffs_delta', 'cluster_bootstrap', 'cohens_d',
    'compute_summary_statistics', 'holm_bonferroni', 'mixed_effects_model_interface',
    'paired_ttest', 'wilcoxon_test',

    # Failure taxonomy
    'compute_failure_rates_by_method', 'compute_failure_rates_by_scenario',

    # Report tables
    'build_ablation_comparison_table', 'build_core_metrics_table',
    'build_failure_taxonomy_table', 'build_runtime_summary_table',
    'build_sensitivity_summary_table', 'save_all_tables',
    'to_latex_table', 'save_all_tables_latex',

    # Basic plots
    'plot_failure_distribution', 'plot_metric_comparison', 'plot_trajectory',

    # Publication-quality plots (pub_plots.py)
    'plot_cpa_evolution', 'plot_control_inputs', 'plot_sensitivity_tornado',
    'plot_radar_chart', 'plot_failure_heatmap', 'plot_trajectory_snapshots',
    'plot_metric_boxplot_swarm', 'plot_violin_with_significance',
    'plot_cdf_comparison', 'plot_forest_effect_sizes', 'plot_mpc_convergence',
    'plot_safety_breakdown_stacked', 'plot_colregs_compliance_by_encounter',
    'plot_parameter_sensitivity_heatmap', 'plot_kinematic_phase_portrait',
    'plot_encounter_geometry', 'plot_figure1_overview',
    'plot_figure2_statistical_comparison',
    'journal_style', 'add_significance_brackets',
]
