#!/usr/bin/env python3
"""
Demo script showing how easy it is to switch between different feature sets
for clustering analysis with the new modular system.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from feature_definitions import FeatureManager, list_available_feature_sets
from flexible_clusterer import FlexibleEcosystemClusterer

def demo_feature_switching():
    """Demonstrate the flexibility of the new feature selection system"""
    
    print("🧬 FEATURE SWITCHING DEMO")
    print("=" * 50)
    
    # Show all available feature sets
    print("\n📊 Step 1: List all available feature sets")
    list_available_feature_sets()
    
    # Create a feature manager to explore features
    print("\n🔍 Step 2: Explore feature set details")
    manager = FeatureManager()
    
    # Show details for a few feature sets
    for feature_set_name in ['geographic', 'climate', 'comprehensive']:
        manager.print_feature_set_summary(feature_set_name)
    
    # Demonstrate easy feature set switching
    print("\n🔄 Step 3: Demonstrate easy feature switching")
    
    # Create clusterer with one feature set
    print("\n  🎯 Create clusterer with 'geographic' features:")
    clusterer = FlexibleEcosystemClusterer(
        feature_set_name='geographic',
        verbose=True
    )
    
    # Switch to different feature sets
    print("\n  🔄 Switch to 'climate' features:")
    clusterer.switch_feature_set('climate')
    
    print("\n  🔄 Switch to 'comprehensive' features:")
    clusterer.switch_feature_set('comprehensive')
    
    # Show how to create custom feature sets
    print("\n🛠️  Step 4: Create custom feature sets")
    
    custom_features = manager.create_custom_feature_set(
        name='minimal_demo',
        numeric_features=['longitude', 'latitude', 'elevation', 'mean_annual_temp'],
        categorical_features=['biome_code'],
        description='Minimal demo feature set for testing'
    )
    
    print(f"✅ Created custom feature set: {custom_features.name}")
    print(f"📊 Features: {custom_features.feature_count} total")
    
    # Switch to custom feature set
    clusterer.switch_feature_set('minimal_demo')

def demo_cli_usage():
    """Show examples of CLI usage"""
    
    print("\n\n🖥️  CLI USAGE EXAMPLES")
    print("=" * 50)
    
    examples = [
        {
            'title': 'Basic geographic clustering',
            'command': 'python FlexibleClusteringPipeline.py --feature-set geographic'
        },
        {
            'title': 'Climate-focused clustering',
            'command': 'python FlexibleClusteringPipeline.py --feature-set climate --clusters 3,4,5'
        },
        {
            'title': 'Comprehensive analysis with train/test split',
            'command': 'python FlexibleClusteringPipeline.py --feature-set comprehensive --site-split-file ../models/site_split.json'
        },
        {
            'title': 'List all available feature sets',
            'command': 'python FlexibleClusteringPipeline.py --list-feature-sets'
        },
        {
            'title': 'Analyze feature compatibility without clustering',
            'command': 'python FlexibleClusteringPipeline.py --analyze-only --feature-set climate'
        },
        {
            'title': 'Custom clustering parameters',
            'command': 'python FlexibleClusteringPipeline.py --feature-set ecological --missing-strategy mean --min-availability 0.7'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print(f"   {example['command']}")

def show_comparison_with_original():
    """Show the improvement over the original clustering_v3.py"""
    
    print("\n\n🔄 COMPARISON: Original vs Modular")
    print("=" * 50)
    
    print("📋 ORIGINAL clustering_v3.py:")
    print("  ❌ Hardcoded feature lists in __init__")
    print("  ❌ Only 'hybrid' feature set supported")
    print("  ❌ Must modify code to change features")
    print("  ❌ Monolithic class with mixed responsibilities")
    
    print("\n📋 NEW Modular System:")
    print("  ✅ Predefined feature sets: geographic, climate, ecological, comprehensive, performance, environmental")
    print("  ✅ Easy feature switching: clusterer.switch_feature_set('climate')")
    print("  ✅ Custom feature sets: manager.create_custom_feature_set(...)")
    print("  ✅ CLI feature selection: --feature-set climate")
    print("  ✅ Feature compatibility analysis")
    print("  ✅ Modular components: FeatureManager, DataPreprocessor, FlexibleClusterer")
    print("  ✅ Same clustering algorithms and quality")
    
    print("\n🎯 USAGE COMPARISON:")
    
    print("\n  Original (clustering_v3.py):")
    print("    # To change features, edit lines 46-59 in code:")
    print("    self.hybrid_numeric = ['longitude', 'latitude', ...]")
    print("    # Then modify prepare_clustering_data() method")
    
    print("\n  Modular (new system):")
    print("    # Command line:")
    print("    python FlexibleClusteringPipeline.py --feature-set climate")
    print("    ")
    print("    # Or programmatically:")
    print("    clusterer = FlexibleEcosystemClusterer(feature_set_name='climate')")
    print("    clusterer.switch_feature_set('geographic')  # Easy switching!")

def main():
    """Run the complete demo"""
    
    try:
        demo_feature_switching()
        demo_cli_usage()
        show_comparison_with_original()
        
        print("\n\n🎉 DEMO COMPLETE!")
        print("=" * 50)
        print("✅ The modular system makes feature selection much easier!")
        print("🚀 Try running: python FlexibleClusteringPipeline.py --list-feature-sets")
        print("💡 Or start with: python FlexibleClusteringPipeline.py --feature-set geographic --analyze-only")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running this from the ecosystem/clustering/ directory")
        print("💡 Or ensure the clustering modules are in your Python path")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()