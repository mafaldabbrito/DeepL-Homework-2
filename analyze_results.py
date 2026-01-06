"""
Analysis script to evaluate validation correlation vs test correlation.

This helps identify:
- Overfitting (val corr much higher than test)
- Underfitting (val corr plateaus far from test)
- Optimal stopping point (where val corr is closest to test)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple


def load_results(protein_name: str, results_dir: str = "results") -> Dict:
    """Load training results from JSON."""
    results_path = os.path.join(results_dir, f"{protein_name}_final_results.json")
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_val_vs_test(results: Dict) -> Dict:
    """
    Analyze how close validation correlation gets to test correlation.
    
    Returns:
        Dictionary with analysis metrics
    """
    history = results['history']
    val_corrs = np.array(history['val_corr'])
    test_corr = results['test_correlation']
    
    # Find best validation epoch
    best_epoch = np.argmax(val_corrs)
    best_val_corr = val_corrs[best_epoch]
    
    # Calculate differences
    final_val_corr = val_corrs[-1]
    best_gap = test_corr - best_val_corr
    final_gap = test_corr - final_val_corr
    
    # Find epoch where val corr is closest to test
    abs_diffs = np.abs(val_corrs - test_corr)
    closest_epoch = np.argmin(abs_diffs)
    closest_val_corr = val_corrs[closest_epoch]
    min_gap = abs_diffs[closest_epoch]
    
    # Check for overfitting indicators
    val_peak = np.max(val_corrs)
    val_final = val_corrs[-1]
    overfitting_drop = val_peak - val_final
    
    analysis = {
        'test_correlation': test_corr,
        'best_val_correlation': best_val_corr,
        'best_val_epoch': int(best_epoch + 1),
        'final_val_correlation': final_val_corr,
        'final_epoch': len(val_corrs),
        'closest_val_correlation': closest_val_corr,
        'closest_epoch': int(closest_epoch + 1),
        'best_gap': best_gap,
        'final_gap': final_gap,
        'min_gap': min_gap,
        'overfitting_drop': overfitting_drop,
        'relative_error_best': abs(best_gap / test_corr) * 100,
        'relative_error_final': abs(final_gap / test_corr) * 100,
        'relative_error_closest': (min_gap / test_corr) * 100
    }
    
    return analysis


def print_analysis(analysis: Dict, protein_name: str):
    """Print formatted analysis report."""
    print(f"\n{'='*70}")
    print(f"VALIDATION vs TEST CORRELATION ANALYSIS - {protein_name}")
    print(f"{'='*70}\n")
    
    print("Test Set Performance:")
    print(f"  Test Correlation: {analysis['test_correlation']:.4f}")
    print()
    
    print("Validation Set Performance:")
    print(f"  Best Val Correlation:  {analysis['best_val_correlation']:.4f} (Epoch {analysis['best_val_epoch']})")
    print(f"  Final Val Correlation: {analysis['final_val_correlation']:.4f} (Epoch {analysis['final_epoch']})")
    print(f"  Closest to Test:       {analysis['closest_val_correlation']:.4f} (Epoch {analysis['closest_epoch']})")
    print()
    
    print("Gap Analysis (Val - Test):")
    print(f"  Gap at Best Val:    {analysis['best_gap']:+.4f} ({analysis['relative_error_best']:.2f}% relative error)")
    print(f"  Gap at Final Epoch: {analysis['final_gap']:+.4f} ({analysis['relative_error_final']:.2f}% relative error)")
    print(f"  Minimum Gap:        {analysis['min_gap']:+.4f} ({analysis['relative_error_closest']:.2f}% relative error)")
    print()
    
    print("Overfitting Indicators:")
    print(f"  Val Peak-to-Final Drop: {analysis['overfitting_drop']:.4f}")
    if analysis['overfitting_drop'] > 0.01:
        print("  ⚠️  Warning: Significant drop suggests overfitting")
    elif analysis['overfitting_drop'] < 0:
        print("  ✓ Still improving at end of training")
    else:
        print("  ✓ Minimal overfitting")
    print()
    
    print("Interpretation:")
    if abs(analysis['best_gap']) < 0.01:
        print("  ✓ Excellent: Val correlation very close to test performance")
    elif abs(analysis['best_gap']) < 0.03:
        print("  ✓ Good: Val correlation reasonably predicts test performance")
    elif analysis['best_gap'] < 0:
        print("  ⚠️  Val correlation exceeds test (possible overfitting to val set)")
    else:
        print("  ⚠️  Val correlation underestimates test (consider more training)")
    
    print(f"\n{'='*70}\n")


def plot_correlation_comparison(results: Dict, analysis: Dict, protein_name: str, 
                                 output_dir: str = "results"):
    """Create detailed plot comparing val and test correlations."""
    history = results['history']
    epochs = np.arange(1, len(history['val_corr']) + 1)
    val_corrs = np.array(history['val_corr'])
    train_corrs = np.array(history['train_corr'])
    test_corr = results['test_correlation']
    
    plt.figure(figsize=(12, 6))
    
    # Main plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_corrs, 'b-', label='Train', alpha=0.7, linewidth=2)
    plt.plot(epochs, val_corrs, 'g-', label='Validation', alpha=0.7, linewidth=2)
    plt.axhline(y=test_corr, color='r', linestyle='--', label='Test (final)', linewidth=2)
    
    # Mark best val epoch
    best_epoch = analysis['best_val_epoch']
    best_val = analysis['best_val_correlation']
    plt.scatter([best_epoch], [best_val], color='green', s=100, zorder=5, 
                marker='o', edgecolors='black', linewidths=2)
    
    # Mark closest to test
    closest_epoch = analysis['closest_epoch']
    closest_val = analysis['closest_val_correlation']
    plt.scatter([closest_epoch], [closest_val], color='orange', s=100, zorder=5,
                marker='*', edgecolors='black', linewidths=2)
    
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Spearman Correlation', fontsize=11)
    plt.title(f'{protein_name}: Correlation Progression', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim([0.35, max(train_corrs.max(), test_corr) + 0.05])
    
    # Gap plot
    plt.subplot(1, 2, 2)
    gaps = val_corrs - test_corr
    plt.plot(epochs, gaps, 'purple', linewidth=2, label='Val - Test Gap')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Perfect Match')
    plt.fill_between(epochs, gaps, 0, where=(gaps >= 0), alpha=0.3, color='green', label='Val > Test')
    plt.fill_between(epochs, gaps, 0, where=(gaps < 0), alpha=0.3, color='red', label='Val < Test')
    
    # Mark minimum gap
    plt.scatter([closest_epoch], [gaps[closest_epoch-1]], color='orange', s=100, 
                zorder=5, marker='*', edgecolors='black', linewidths=2)
    
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Correlation Gap (Val - Test)', fontsize=11)
    plt.title('Validation-Test Gap Analysis', fontsize=12, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{protein_name}_val_test_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Analysis plot saved to {output_path}")
    plt.close()


def create_summary_table(analysis: Dict, output_dir: str = "results", 
                         protein_name: str = "RBFOX1"):
    """Create a summary table in JSON format."""
    summary = {
        'protein': protein_name,
        'test_correlation': round(analysis['test_correlation'], 4),
        'validation_metrics': {
            'best': {
                'correlation': round(analysis['best_val_correlation'], 4),
                'epoch': analysis['best_val_epoch'],
                'gap_from_test': round(analysis['best_gap'], 4),
                'relative_error_pct': round(analysis['relative_error_best'], 2)
            },
            'final': {
                'correlation': round(analysis['final_val_correlation'], 4),
                'epoch': analysis['final_epoch'],
                'gap_from_test': round(analysis['final_gap'], 4),
                'relative_error_pct': round(analysis['relative_error_final'], 2)
            },
            'closest_to_test': {
                'correlation': round(analysis['closest_val_correlation'], 4),
                'epoch': analysis['closest_epoch'],
                'gap_from_test': round(analysis['min_gap'], 4),
                'relative_error_pct': round(analysis['relative_error_closest'], 2)
            }
        },
        'overfitting_indicator': round(analysis['overfitting_drop'], 4)
    }
    
    output_path = os.path.join(output_dir, f"{protein_name}_val_test_summary.json")
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary table saved to {output_path}")


def main():
    """Run complete validation vs test analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze validation vs test correlation")
    parser.add_argument("--protein", default="RBFOX1", help="Protein name to analyze")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results for {args.protein}...")
    results = load_results(args.protein, args.results_dir)
    
    # Analyze
    analysis = analyze_val_vs_test(results)
    
    # Print report
    print_analysis(analysis, args.protein)
    
    # Create visualizations
    plot_correlation_comparison(results, analysis, args.protein, args.results_dir)
    
    # Create summary table
    create_summary_table(analysis, args.results_dir, args.protein)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
