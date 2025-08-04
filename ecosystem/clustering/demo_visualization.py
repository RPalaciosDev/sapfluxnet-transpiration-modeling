#!/usr/bin/env python3
"""
Demonstration of the clustering visualization capabilities.

Shows how to integrate visualizations into the flexible clustering workflow
for comprehensive analysis and result validation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from flexible_clusterer import FlexibleEcosystemClusterer
    from clustering_visualizer import ClusteringVisualizer
    from feature_definitions import FeatureManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running this from the ecosystem/clustering/ directory")
    sys.exit(1)


def create_sample_data(n_sites=60):
    """Create realistic sample data for demonstration"""
    np.random.seed(42)
    
    # Create three natural geographic clusters
    cluster_centers = [
        {'lat': 45.0, 'lon': -120.0, 'temp': 10.0},  # Pacific Northwest
        {'lat': 35.0, 'lon': -105.0, 'temp': 15.0},  # Southwest
        {'lat': 48.0, 'lon': -95.0, 'temp': 5.0}     # Upper Midwest
    ]
    
    sites_data = []
    for i, center in enumerate(cluster_centers):
        n_cluster_sites = n_sites // 3 + (1 if i < n_sites % 3 else 0)
        
        for j in range(n_cluster_sites):
            site_data = {
                'site': f'site_{i+1}_{j+1:02d}',
                'longitude': np.random.normal(center['lon'], 2.0),
                'latitude': np.random.normal(center['lat'], 1.5),
                'elevation': np.random.uniform(100, 2000),
                'mean_annual_temp': np.random.normal(center['temp'], 3.0),
                'mean_annual_precip': np.random.uniform(300, 1500),
                'seasonal_temp_range': np.random.uniform(10, 30),
                'seasonal_precip_range': np.random.uniform(100, 600),
                'basal_area': np.random.uniform(10, 50),
                'tree_density': np.random.uniform(100, 800),
                'leaf_area_index': np.random.uniform(1, 8),
                'biome_code': np.random.choice(['temperate_forest', 'mixed_forest', 'boreal_forest']),
                'igbp_class_code': np.random.choice(['ENF', 'DBF', 'MF'])
            }
            sites_data.append(site_data)
    
    return pd.DataFrame(sites_data)


def demo_basic_visualization():
    """Demonstrate basic visualization workflow"""
    print("🎨 BASIC VISUALIZATION DEMO")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample ecological data...")
    sample_data = create_sample_data(n_sites=45)
    
    # Create visualizer directly
    print("\n🖼️  Demo 1: Direct Visualizer Usage")
    visualizer = ClusteringVisualizer(output_dir='demo_visualizations')
    
    # Simulate clustering results
    features = ['longitude', 'latitude', 'elevation', 'mean_annual_temp', 'mean_annual_precip']
    
    # Simple K-means for demo
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    X = sample_data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    strategy_info = {
        'name': 'K-means (3 clusters)',
        'algorithm': 'K-means',
        'silhouette': 0.543,
        'balance_ratio': 0.667
    }
    
    # Set up visualizer
    visualizer.set_clustering_data(sample_data, features, labels, strategy_info)
    
    # Generate individual visualizations
    print("  📈 Generating PCA plot...")
    visualizer.plot_feature_space_2d(method='pca', show=False)
    
    print("  📈 Generating t-SNE plot...")
    visualizer.plot_feature_space_2d(method='tsne', show=False)
    
    print("  📈 Generating silhouette analysis...")
    visualizer.plot_silhouette_analysis(show=False)
    
    print("  🗺️  Generating geographic plot...")
    visualizer.plot_geographic_clusters(show=False)
    
    print("  📊 Generating interactive dashboard...")
    visualizer.create_interactive_dashboard()
    
    print("✅ Basic visualization demo complete!")


def demo_integrated_workflow():
    """Demonstrate integrated clustering + visualization workflow"""
    print("\n🚀 INTEGRATED WORKFLOW DEMO")
    print("=" * 50)
    
    # Create sample data
    sample_data = create_sample_data(n_sites=50)
    
    # Save sample data as if it were processed parquet files
    demo_data_dir = Path('demo_data')
    demo_data_dir.mkdir(exist_ok=True)
    
    # Split data into individual site files (simulating processed parquet structure)
    for site in sample_data['site'].unique()[:10]:  # Use subset for demo
        site_data = sample_data[sample_data['site'] == site]
        # Duplicate rows to simulate time series data
        site_data_expanded = pd.concat([site_data] * 24, ignore_index=True)  # 24 time points
        site_data_expanded.to_parquet(demo_data_dir / f'{site}_comprehensive.parquet')
    
    print(f"📁 Created demo data in: {demo_data_dir}")
    
    # Demo 2: Integrated clustering with visualization
    print("\n🧬 Demo 2: Integrated Clustering + Visualization")
    
    try:
        # Create clusterer
        clusterer = FlexibleEcosystemClusterer(
            data_dir=str(demo_data_dir),
            output_dir='demo_clustering_results',
            feature_set_name='climate',
            verbose=True
        )
        
        # Run clustering
        print("🎯 Running clustering...")
        output_file = clusterer.run_clustering(cluster_range=[2, 3, 4])
        
        if output_file:
            print("\n🎨 Generating visualizations...")
            
            # Quick visualization
            print("  📊 Quick PCA visualization...")
            clusterer.quick_visualize(method='pca', show=False)
            
            # Interactive dashboard
            print("  📊 Interactive dashboard...")
            clusterer.create_interactive_dashboard()
            
            # Comprehensive visualizations
            print("  📊 Comprehensive visualization report...")
            visualizations = clusterer.visualize_clustering(
                strategies=getattr(clusterer, '_all_strategies', None),
                include_3d=True,
                include_dashboard=True,
                show_plots=False
            )
            
            print(f"✅ Generated {len(visualizations)} visualizations:")
            for name, path in visualizations.items():
                print(f"    🎨 {name}: {Path(path).name}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup demo data
        import shutil
        if demo_data_dir.exists():
            shutil.rmtree(demo_data_dir)
        print(f"🧹 Cleaned up demo data directory")


def demo_cli_visualization():
    """Demonstrate CLI visualization options"""
    print("\n🖥️  CLI VISUALIZATION DEMO")
    print("=" * 50)
    
    print("📋 Available CLI visualization options:")
    print()
    
    examples = [
        {
            'title': 'Quick PCA visualization',
            'command': 'python FlexibleClusteringPipeline.py --feature-set climate --quick-viz pca'
        },
        {
            'title': 'Quick geographic visualization',
            'command': 'python FlexibleClusteringPipeline.py --feature-set climate --quick-viz geographic'
        },
        {
            'title': 'Interactive dashboard only',
            'command': 'python FlexibleClusteringPipeline.py --feature-set climate --dashboard'
        },
        {
            'title': 'Comprehensive visualizations (with 3D)',
            'command': 'python FlexibleClusteringPipeline.py --feature-set climate --visualize'
        },
        {
            'title': 'Comprehensive visualizations (no 3D, faster)',
            'command': 'python FlexibleClusteringPipeline.py --feature-set climate --visualize --no-3d'
        },
        {
            'title': 'Clustering + all visualizations',
            'command': 'python FlexibleClusteringPipeline.py --feature-set comprehensive --visualize --dashboard --quick-viz pca'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['title']}:")
        print(f"   {example['command']}")
        print()


def demo_programmatic_usage():
    """Show programmatic usage examples"""
    print("\n💻 PROGRAMMATIC USAGE EXAMPLES")
    print("=" * 50)
    
    code_examples = [
        {
            'title': 'Basic clustering with quick visualization',
            'code': """
# Basic workflow
clusterer = FlexibleEcosystemClusterer(feature_set_name='climate')
clusterer.run_clustering()
clusterer.quick_visualize(method='pca')
            """
        },
        {
            'title': 'Feature comparison with visualizations',
            'code': """
# Compare different feature sets
    for feature_set in ['geographic', 'biome', 'climate', 'comprehensive', 'plant_functional', 'v2_hybrid', 'v3_hybrid']:
    clusterer = FlexibleEcosystemClusterer(feature_set_name=feature_set)
    clusterer.run_clustering()
    clusterer.quick_visualize(method='pca', show=False)
    print(f"Results for {feature_set} features saved")
            """
        },
        {
            'title': 'Comprehensive analysis with all visualizations',
            'code': """
# Full analysis workflow
clusterer = FlexibleEcosystemClusterer(feature_set_name='comprehensive')
clusterer.run_clustering()

# Generate all visualizations
visualizations = clusterer.visualize_clustering(
    include_3d=True,
    include_dashboard=True
)

# Also create interactive dashboard
dashboard = clusterer.create_interactive_dashboard()
            """
        },
        {
            'title': 'Custom visualization workflow',
            'code': """
# Custom visualization workflow
clusterer = FlexibleEcosystemClusterer(feature_set_name='climate')
clusterer.run_clustering()

# Access the visualizer directly for custom plots
visualizer = clusterer.visualizer
visualizer.plot_feature_space_2d(method='pca')
visualizer.plot_feature_space_3d_interactive(method='tsne')
visualizer.plot_silhouette_analysis()
            """
        }
    ]
    
    for i, example in enumerate(code_examples, 1):
        print(f"{i}. {example['title']}:")
        print(example['code'])
        print()


def main():
    """Run the complete visualization demonstration"""
    print("🎨 CLUSTERING VISUALIZATION DEMONSTRATION")
    print("=" * 60)
    print("This demo shows how to use the new visualization capabilities")
    print("integrated with the flexible clustering system.")
    print()
    
    try:
        # Check dependencies
        print("🔍 Checking visualization dependencies...")
        missing_deps = []
        
        try:
            import matplotlib
            print("  ✅ matplotlib available")
        except ImportError:
            missing_deps.append("matplotlib")
        
        try:
            import seaborn
            print("  ✅ seaborn available")
        except ImportError:
            missing_deps.append("seaborn")
        
        try:
            import plotly
            print("  ✅ plotly available")
        except ImportError:
            missing_deps.append("plotly")
        
        try:
            from sklearn.decomposition import PCA
            print("  ✅ scikit-learn available")
        except ImportError:
            missing_deps.append("scikit-learn")
        
        if missing_deps:
            print(f"\n⚠️  Missing dependencies: {missing_deps}")
            print("💡 Install with: pip install matplotlib seaborn plotly scikit-learn")
            print("   Some demos may not work without these packages.")
            print()
        
        # Run demos
        demo_basic_visualization()
        demo_integrated_workflow()
        demo_cli_visualization()
        demo_programmatic_usage()
        
        print("\n🎉 VISUALIZATION DEMO COMPLETE!")
        print("=" * 60)
        print("✅ All visualization capabilities demonstrated")
        print("📁 Check the demo_visualizations/ directory for sample outputs")
        print("💡 Try the CLI examples to see visualizations with your data")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()