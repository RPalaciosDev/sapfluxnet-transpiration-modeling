#!/usr/bin/env python3
"""
Flexible Clustering Pipeline - CLI wrapper with easy feature selection

This script provides a clean command-line interface for the modular clustering
pipeline with easy feature set switching and experimentation.
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add the clustering directory to path for imports
sys.path.append(str(Path(__file__).parent))

from flexible_clusterer import FlexibleEcosystemClusterer
from feature_definitions import FeatureManager, list_available_feature_sets
from data_preprocessor import ClusteringDataPreprocessor


def main():
    """Main execution function with enhanced CLI"""
    
    parser = argparse.ArgumentParser(
        description='Flexible Ecosystem Clustering with Easy Feature Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic geographic clustering
  python FlexibleClusteringPipeline.py --feature-set geographic
  
  # Climate-focused clustering with PCA visualization  
  python FlexibleClusteringPipeline.py --feature-set climate --quick-viz pca
  
  # Comprehensive clustering with interactive dashboard
  python FlexibleClusteringPipeline.py --feature-set comprehensive --dashboard
  
  # Full analysis with all visualizations
  python FlexibleClusteringPipeline.py --feature-set climate --visualize
  
  # Fast analysis (no 3D visualizations)
  python FlexibleClusteringPipeline.py --feature-set climate --visualize --no-3d
  
  # List all available feature sets
  python FlexibleClusteringPipeline.py --list-feature-sets
  
  # Analyze feature compatibility without clustering
  python FlexibleClusteringPipeline.py --analyze-only --feature-set climate
        """
    )
    
    # Data directories
    parser.add_argument('--data-dir', default='../../processed_parquet',
                       help='Directory containing processed parquet files (default: ../../processed_parquet)')
    parser.add_argument('--output-dir', default='../evaluation/clustering_results',
                       help='Base directory for clustering results. Auto-creates subdirectories with pattern feature-set_date (default: ../evaluation/clustering_results)')
    
    # Feature selection (the main improvement!)
    parser.add_argument('--feature-set',
                        choices=['geographic', 'biome', 'climate', 'ecological', 'comprehensive', 'performance', 'environmental', 'plant_functional', 'v2_core', 'v2_advanced', 'v2_hybrid', 'v3_hybrid', 'advanced_core', 'advanced_derived', 'advanced_hybrid'],
                       default='comprehensive',
                       help='Feature set to use for clustering (default: comprehensive)')
    
    # Clustering parameters
    parser.add_argument('--clusters', type=str, default='3,4,5,6,7,8,9,10',
                       help='Comma-separated list of cluster numbers to try (default: 3,4,5,6,7,8,9,10)')
    parser.add_argument('--missing-strategy', choices=['median', 'mean', 'drop', 'zero'], default='median',
                       help='Strategy for handling missing values (default: median)')
    parser.add_argument('--min-availability', type=float, default=0.5,
                       help='Minimum feature availability ratio to proceed (default: 0.5)')
    
    # Strategy selection parameters
    parser.add_argument('--min-balance', type=float, default=0.15,
                       help='Minimum cluster balance ratio to consider (default: 0.15)')
    parser.add_argument('--silhouette-weight', type=float, default=1.5,
                       help='Weight for silhouette score in strategy selection (default: 1.5)')
    parser.add_argument('--balance-weight', type=float, default=1.0,
                       help='Weight for balance ratio in strategy selection (default: 1.0)')
    
    # Site split and validation
    parser.add_argument('--site-split-file', 
                       help='JSON file with train/test site split (optional)')
    
    # Analysis and information options
    parser.add_argument('--list-feature-sets', action='store_true',
                       help='List all available feature sets and exit')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze feature compatibility without clustering')
    parser.add_argument('--show-feature-details', action='store_true',
                       help='Show detailed information about the selected feature set')
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                       help='Generate comprehensive visualizations after clustering')
    parser.add_argument('--quick-viz', choices=['pca', 'tsne', 'geographic', 'silhouette'],
                       help='Generate a quick visualization of specified type')
    parser.add_argument('--dashboard', action='store_true',
                       help='Create interactive dashboard')
    parser.add_argument('--no-3d', action='store_true',
                       help='Skip 3D interactive visualizations (faster)')
    
    # Output options
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output (default: True)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Handle quiet mode
    verbose = args.verbose and not args.quiet
    
    # Print header
    if verbose:
        print("*** FLEXIBLE ECOSYSTEM CLUSTERING PIPELINE ***")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
    
    # Handle list feature sets
    if args.list_feature_sets:
        print("\n>> AVAILABLE FEATURE SETS:")
        print("=" * 40)
        list_available_feature_sets()
        
        # Also show compatibility analysis if data directory exists
        if Path(args.data_dir).exists():
            print(f"\n💡 To analyze feature compatibility with your data:")
            print(f"python {Path(__file__).name} --analyze-only --feature-set <feature_set_name>")
        
        return True
    
    # Validate and parse cluster range
    try:
        cluster_range = [int(x.strip()) for x in args.clusters.split(',')]
        cluster_range = [c for c in cluster_range if c >= 2]  # Ensure valid cluster counts
        if not cluster_range:
            raise ValueError("No valid cluster numbers provided")
    except ValueError as e:
        print(f"ERROR: Invalid cluster range '{args.clusters}': {e}")
        print("TIP: Use format like: 3,4,5,6 or 2,3,4,5,6,7,8,9,10")
        return False
    
    if verbose:
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f">> Feature set: {args.feature_set}")
        print(f"Cluster range: {cluster_range}")
        print(f"Missing value strategy: {args.missing_strategy}")
        print(f"Minimum feature availability: {args.min_availability:.1%}")
        
        if args.site_split_file:
            print(f"Site split file: {args.site_split_file}")
        else:
            print(f"WARNING: No site split - clustering all sites")
    
    try:
        # Create dynamic output directory with pattern: feature-set_date
        from datetime import datetime as dt
        from pathlib import Path
        
        timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
        dynamic_output_dir = Path(args.output_dir) / f"{args.feature_set}_{timestamp}"
        dynamic_output_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"Dynamic output directory: {dynamic_output_dir}")
        
        # Create the flexible clusterer
        clusterer = FlexibleEcosystemClusterer(
            data_dir=args.data_dir,
            output_dir=str(dynamic_output_dir),
            feature_set_name=args.feature_set,
            site_split_file=args.site_split_file,
            verbose=verbose
        )
        
        # Show feature set details if requested
        if args.show_feature_details:
            clusterer.feature_manager.print_feature_set_summary(args.feature_set)
        
        # Handle analyze-only mode
        if args.analyze_only:
            if verbose:
                print(f"\n🔍 FEATURE COMPATIBILITY ANALYSIS ONLY")
                print(f"Loading data to analyze feature availability...")
            
            # Load data and analyze compatibility
            site_df = clusterer.load_site_data()
            compatibility = clusterer.preprocessor.analyze_data_compatibility(site_df, args.feature_set)
            
            # Suggest alternatives if availability is low
            if compatibility['availability_ratio'] < args.min_availability:
                print(f"\n💡 SUGGESTED ALTERNATIVES (≥{args.min_availability:.0%} availability):")
                recommendations = clusterer.preprocessor.suggest_best_feature_sets(site_df, args.min_availability)
                
                if recommendations:
                    for i, rec in enumerate(recommendations[:5], 1):
                        print(f"  {i}. {rec['name']}: {rec['availability_ratio']:.1%} availability ({rec['available_features']} features)")
                        print(f"     {rec['description']}")
                else:
                    print(f"  ❌ No feature sets meet minimum availability of {args.min_availability:.0%}")
            else:
                print(f"\n✅ Feature set '{args.feature_set}' is compatible with your data!")
                print(f"🎯 Ready to proceed with clustering using {compatibility['total_available']} features")
            
            return True
        
        # Run the clustering pipeline
        if verbose:
            print(f"\n🚀 Starting clustering pipeline...")
        
        output_file = clusterer.run_clustering(
            missing_strategy=args.missing_strategy,
            min_availability=args.min_availability,
            cluster_range=cluster_range
        )
        
        # Generate visualizations if requested
        # Always generate visualizations when clustering is successful
        if output_file:
            if verbose:
                print(f"\n🎨 Generating visualizations...")
            
            try:
                # Always generate comprehensive visualizations
                strategies = getattr(clusterer, '_all_strategies', None)
                
                visualizations = clusterer.visualize_clustering(
                    strategies=strategies,
                    include_3d=not args.no_3d if hasattr(args, 'no_3d') else True,
                    include_dashboard=True,
                    show_plots=False
                )
                
                if verbose:
                    print(f"📊 Generated {len(visualizations)} comprehensive visualizations")
                
                # Also generate quick visualization if specified
                if args.quick_viz:
                    viz_file = clusterer.quick_visualize(method=args.quick_viz, show=False)
                    if verbose:
                        print(f"📊 Quick visualization ({args.quick_viz}): {Path(viz_file).name}")
                
                # Also generate interactive dashboard if specified
                if args.dashboard:
                    dashboard_file = clusterer.create_interactive_dashboard()
                    if verbose:
                        print(f"📊 Interactive dashboard: {Path(dashboard_file).name}")
                        
            except ImportError as e:
                print(f"⚠️  Visualization libraries not available: {e}")
                print("💡 Install with: pip install matplotlib seaborn plotly scikit-learn")
            except Exception as e:
                print(f"⚠️  Visualization generation failed: {e}")
        
        if output_file:
            if verbose:
                print(f"\n🎉 CLUSTERING COMPLETED SUCCESSFULLY!")
                print(f"📁 Results saved to: {output_file}")
                
                # Show summary of results
                if clusterer.clustering_results:
                    cluster_counts = clusterer.clustering_results['cluster_counts']
                    print(f"\n📊 FINAL CLUSTER SUMMARY:")
                    total_sites = cluster_counts.sum()
                    for cluster_id, count in cluster_counts.items():
                        percentage = (count / total_sites) * 100
                        print(f"  Cluster {cluster_id}: {count:3d} sites ({percentage:5.1f}%)")
                
                # Provide next steps guidance
                print(f"\n🎯 NEXT STEPS:")
                print(f"  1. Review cluster assignments: {output_file}")
                print(f"  2. Examine strategy details: {clusterer.clustering_results['strategy_file']}")
                
                if args.site_split_file:
                    print(f"  3. Use clusters for ensemble model training")
                    print(f"  4. Evaluate spatial validation performance")
                else:
                    print(f"  3. Create train/test split for model validation")
                    print(f"  4. Use clusters for ensemble modeling")
                
                # Feature set switching guidance
                print(f"\n💡 FEATURE EXPERIMENTATION:")
                print(f"  Try different feature sets to compare clustering results:")
                current_set = args.feature_set
                other_sets = [s for s in ['geographic', 'biome', 'climate', 'ecological', 'comprehensive', 'plant_functional', 'v2_hybrid', 'v3_hybrid', 'advanced_hybrid'] if s != current_set]
                for alt_set in other_sets[:2]:  # Show 2 alternatives
                    print(f"  python {Path(__file__).name} --feature-set {alt_set} --data-dir {args.data_dir}")
            
            return True
        else:
            print(f"\n❌ Clustering failed - no output file generated")
            return False
            
    except KeyboardInterrupt:
        print(f"\n🛑 Clustering interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Clustering failed with error: {str(e)}")
        if verbose:
            import traceback
            print("\n📋 Full error trace:")
            traceback.print_exc()
        return False
    
    finally:
        if verbose:
            print(f"\n⏰ Pipeline finished at: {datetime.now()}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)