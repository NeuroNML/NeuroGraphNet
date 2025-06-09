#!/usr/bin/env python3
"""
Metrics Reporter for Neural Network Training Results

This module provides utilities to extract and format comprehensive metrics
from k-fold cross-validation results for research reports.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


def load_kfold_results(summary_path: Path) -> Dict[str, Any]:
    """Load k-fold cross-validation results from JSON file."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def extract_fold_metrics(fold_results: List[Dict[str, Any]], metric_name: str = 'val_history') -> Dict[str, List[float]]:
    """
    Extract metrics from individual fold results.
    
    Args:
        fold_results: List of fold result dictionaries
        metric_name: Either 'val_history' or 'train_history'
        
    Returns:
        Dictionary with metric names as keys and lists of best values per fold
    """
    metrics_per_fold = {}
    
    for fold in fold_results:
        if 'error' in fold:
            continue  # Skip failed folds
            
        history = fold.get(metric_name, {})
        for metric, values in history.items():
            if metric not in metrics_per_fold:
                metrics_per_fold[metric] = []
            
            if values:
                # Get the best value for this metric
                if metric == 'loss':
                    best_val = min(values)
                else:
                    best_val = max(values)
                metrics_per_fold[metric].append(best_val)
    
    return metrics_per_fold


def calculate_statistics(values: List[float]) -> Dict[str, floating[Any] | List[float]]:
    """Calculate comprehensive statistics for a list of values."""
    if not values:
        return {}
    
    values_array = np.array(values)
    return {
        'mean': np.mean(values_array),
        'std': np.std(values_array),
        'median': np.median(values_array),
        'q25': np.percentile(values_array, 25),
        'q75': np.percentile(values_array, 75),
        'iqr': np.percentile(values_array, 75) - np.percentile(values_array, 25),
        'min': np.min(values_array),
        'max': np.max(values_array),
        'values': values
    }


def format_metric_report(metric_name: str, stats: Dict[str, np.floating[Any] | float], precision: int = 4) -> str:
    """Format a single metric's statistics for reporting."""
    if not stats:
        return f"{metric_name}: No data available"
    
    mean_std = f"{stats['mean']:.{precision}f} ± {stats['std']:.{precision}f}"
    median_iqr = f"Med: {stats['median']:.{precision}f}, IQR: {stats['iqr']:.{precision}f}"
    range_str = f"Range: [{stats['min']:.{precision}f}, {stats['max']:.{precision}f}]"
    
    return f"• {metric_name}: {mean_std} ({median_iqr}) {range_str}"


def generate_comprehensive_report(summary_path: Path, output_format: str = 'text') -> str:
    """
    Generate a comprehensive metrics report from k-fold results.
    
    Args:
        summary_path: Path to the k-fold summary JSON file
        output_format: 'text', 'markdown', or 'latex'
        
    Returns:
        Formatted report string
    """
    try:
        data = load_kfold_results(summary_path)
    except Exception as e:
        return f"Error loading results: {e}"
    
    fold_results = data.get('fold_results', [])
    successful_folds = [f for f in fold_results if 'error' not in f]
    
    if not successful_folds:
        return "No successful folds found in the results."
    
    # Extract validation metrics
    val_metrics = extract_fold_metrics(successful_folds, 'val_history')
    
    # Key metrics for report
    key_metrics = {
        'macro_f1': 'Macro F1-score',
        'auroc': 'AUC', 
        'f1': 'F1 (seizure)',
        'recall': 'Recall',
        'precision': 'Precision'
    }
    
    report_lines = []
    report_lines.append("📋 COMPREHENSIVE METRICS REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"K-Fold Cross-Validation Results ({len(successful_folds)} folds)")
    report_lines.append("")
    
    # Calculate and format each metric
    all_stats = {}
    for metric_key, metric_display in key_metrics.items():
        if metric_key in val_metrics:
            stats = calculate_statistics(val_metrics[metric_key])
            all_stats[metric_key] = stats
            report_lines.append(format_metric_report(metric_display, stats))
    
    # Add detailed fold-by-fold breakdown
    report_lines.append("")
    report_lines.append("📊 FOLD-BY-FOLD BREAKDOWN:")
    report_lines.append("-" * 30)
    
    if val_metrics:
        # Create a DataFrame for better visualization
        fold_data = {}
        for metric in key_metrics.keys():
            if metric in val_metrics:
                fold_data[metric] = val_metrics[metric]
        
        if fold_data:
            df = pd.DataFrame(fold_data)
            df.index = [f"Fold {i+1}" for i in range(len(df))]
            report_lines.append(df.round(4).to_string())
    
    # Summary statistics table
    report_lines.append("")
    report_lines.append("📈 SUMMARY STATISTICS:")
    report_lines.append("-" * 30)
    
    summary_data = []
    for metric_key, metric_display in key_metrics.items():
        if metric_key in all_stats:
            stats = all_stats[metric_key]
            summary_data.append({
                'Metric': metric_display,
                'Mean': f"{stats['mean']:.4f}",
                'Std': f"{stats['std']:.4f}",
                'Median': f"{stats['median']:.4f}",
                'IQR': f"{stats['iqr']:.4f}",
                'Min': f"{stats['min']:.4f}",
                'Max': f"{stats['max']:.4f}"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        report_lines.append(summary_df.to_string(index=False))
    
    # Add LaTeX/Markdown formatting if requested
    if output_format == 'markdown':
        report_lines = ['```'] + report_lines + ['```']
    elif output_format == 'latex':
        # Basic LaTeX formatting
        report_text = '\n'.join(report_lines)
        report_text = report_text.replace('±', '$\\pm$')
        report_lines = ['\\begin{verbatim}'] + [report_text] + ['\\end{verbatim}']
    
    return '\n'.join(report_lines)


def save_metrics_csv(summary_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Save metrics in CSV format for easy import into papers/presentations.
    
    Args:
        summary_path: Path to k-fold summary JSON
        output_path: Where to save CSV (default: same directory as summary)
        
    Returns:
        Path where CSV was saved
    """
    if output_path is None:
        output_path = summary_path.parent / "metrics_summary.csv"
    
    data = load_kfold_results(summary_path)
    fold_results = data.get('fold_results', [])
    successful_folds = [f for f in fold_results if 'error' not in f]
    
    val_metrics = extract_fold_metrics(successful_folds, 'val_history')
    
    # Create CSV data
    csv_data = []
    for metric, values in val_metrics.items():
        if values:
            stats = calculate_statistics(values)
            csv_data.append({
                'metric': metric,
                'mean': stats['mean'],
                'std': stats['std'],
                'median': stats['median'],
                'q25': stats['q25'],
                'q75': stats['q75'],
                'iqr': stats['iqr'],
                'min': stats['min'],
                'max': stats['max'],
                'fold_values': ';'.join(map(str, values))
            })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    
    return output_path


def cli_report(summary_path: str):
    """Command-line interface for generating reports."""
    path = Path(summary_path)
    if not path.exists():
        print(f"Error: File {path} does not exist.")
        return
    
    print(generate_comprehensive_report(path))
    
    # Also save CSV
    csv_path = save_metrics_csv(path)
    print(f"\n💾 Metrics also saved to: {csv_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python metrics_reporter.py <path_to_k_fold_summary.json>")
        sys.exit(1)
    
    cli_report(sys.argv[1])
