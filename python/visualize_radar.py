#!/usr/bin/env python3
"""
Radar data visualization tool
Reads CSV data from radar tests and creates polar plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def create_polar_plot(df, title="Radar Data - Polar View", save_path=None):
    """Create a polar plot of radar data"""
    
    # Create figure with polar projection
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Convert degrees to radians for matplotlib
    theta = np.radians(df['bearing_deg'])
    r = df['range_mm']
    
    # Color points by quality (optional - can use different criteria)
    if 'quality' in df.columns:
        scatter = ax.scatter(theta, r, c=df['quality'], s=20, alpha=0.7, 
                           cmap='viridis', label='LIDAR Points')
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Quality')
    else:
        ax.scatter(theta, r, c='green', s=20, alpha=0.7, label='LIDAR Points')
    
    # Customize the plot
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax.set_theta_zero_location('N')  # 0° at top
    ax.set_theta_direction(1)        # Clockwise
    
    # Set radial limits and labels
    max_range = df['range_mm'].max() if not df.empty else 6000
    ax.set_ylim(0, max_range * 1.1)
    
    # Add range circles every 1000mm
    range_ticks = np.arange(1000, max_range + 1000, 1000)
    ax.set_yticks(range_ticks)
    ax.set_yticklabels([f'{int(r)}mm' for r in range_ticks])
    
    # Add angular grid every 45°
    ax.set_thetagrids(range(0, 360, 45))
    
    # Style the grid
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    
    # Add statistics text
    if not df.empty:
        stats_text = f"Points: {len(df)}\n"
        stats_text += f"Range: {df['range_mm'].min():.0f}-{df['range_mm'].max():.0f}mm\n"
        if 'quality' in df.columns:
            stats_text += f"Quality: {df['quality'].min()}-{df['quality'].max()}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig, ax

def create_cartesian_plot(df, title="Radar Data - Cartesian View", save_path=None):
    """Create a cartesian plot using forward/lateral distances"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'forward_distance' in df.columns and 'lateral_distance' in df.columns:
        x = df['lateral_distance']   # Left/Right
        y = df['forward_distance']   # Forward/Backward
        
        if 'quality' in df.columns:
            scatter = ax.scatter(x, y, c=df['quality'], s=20, alpha=0.7, 
                               cmap='viridis', label='LIDAR Points')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Quality')
        else:
            ax.scatter(x, y, c='green', s=20, alpha=0.7, label='LIDAR Points')
        
        # Mark radar position
        ax.plot(0, 0, 'r*', markersize=15, label='Radar Position')
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Lateral Distance (mm)')
        ax.set_ylabel('Forward Distance (mm)')
        ax.legend()
        
        # Add range circles around radar
        max_range = df['range_mm'].max() if not df.empty else 6000
        for r in range(1000, int(max_range) + 1000, 1000):
            circle = plt.Circle((0, 0), r, fill=False, color='gray', alpha=0.3, linestyle='--')
            ax.add_patch(circle)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig, ax

def analyze_data(df):
    """Print comprehensive data analysis"""
    print("\n" + "="*50)
    print("RADAR DATA ANALYSIS")
    print("="*50)
    
    if df.empty:
        print("No data to analyze!")
        return
    
    print(f"Total Points: {len(df)}")
    print(f"Time Span: {(df['timestamp_ms'].max() - df['timestamp_ms'].min()) / 1000:.1f} seconds")
    
    print(f"\nRange Statistics:")
    print(f"  Min: {df['range_mm'].min():.1f}mm")
    print(f"  Max: {df['range_mm'].max():.1f}mm")
    print(f"  Mean: {df['range_mm'].mean():.1f}mm")
    print(f"  Std: {df['range_mm'].std():.1f}mm")
    
    print(f"\nBearing Statistics:")
    print(f"  Min: {df['bearing_deg'].min():.1f}°")
    print(f"  Max: {df['bearing_deg'].max():.1f}°")
    print(f"  Coverage: {df['bearing_deg'].max() - df['bearing_deg'].min():.1f}° span")
    
    if 'quality' in df.columns:
        print(f"\nQuality Statistics:")
        print(f"  Min: {df['quality'].min()}")
        print(f"  Max: {df['quality'].max()}")
        print(f"  Mean: {df['quality'].mean():.1f}")
    
    # Check for potential issues
    print(f"\nData Quality Checks:")
    zero_range = len(df[df['range_mm'] == 0])
    if zero_range > 0:
        print(f"  ⚠ {zero_range} points with zero range")
    
    low_quality = len(df[df['quality'] < 20]) if 'quality' in df.columns else 0
    if low_quality > 0:
        print(f"  ⚠ {low_quality} points with low quality (<20)")
    
    # Sector coverage analysis
    sectors = np.histogram(df['bearing_deg'], bins=36, range=(0, 360))[0]
    empty_sectors = np.sum(sectors == 0)
    print(f"  Sector coverage: {36 - empty_sectors}/36 sectors have data")
    if empty_sectors > 18:
        print(f"  ⚠ Many empty sectors ({empty_sectors}/36) - limited rotation?")

def main():
    parser = argparse.ArgumentParser(description='Visualize radar data from CSV')
    parser.add_argument('csv_file', nargs='?', default='../src/radar/tests/build/collected_points.csv', 
                       help='CSV file path (default: collected_points.csv)')
    parser.add_argument('--polar', action='store_true', default=True,
                       help='Create polar plot (default)')
    parser.add_argument('--cartesian', action='store_true', 
                       help='Create cartesian plot')
    parser.add_argument('--both', action='store_true',
                       help='Create both polar and cartesian plots')
    parser.add_argument('--save', type=str, default="plots/radar_points",
                       help='Save plots with this prefix')
    parser.add_argument('--no-show', action='store_true',
                       help='Don\'t display plots (useful for batch processing)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found!")
        print("Run the radar test first to generate data.")
        return 1
    
    # Read data
    try:
        df = pd.read_csv(args.csv_file)
        print(f"Loaded {len(df)} points from {args.csv_file}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 1
    
    # Analyze data
    analyze_data(df)
    
    # Create plots based on arguments
    if args.both:
        args.polar = True
        args.cartesian = True
    
    if args.polar:
        save_path = f"{args.save}_polar.png" if args.save else None
        fig_polar, ax_polar = create_polar_plot(df, save_path=save_path)
    
    if args.cartesian:
        save_path = f"{args.save}_cartesian.png" if args.save else None
        fig_cart, ax_cart = create_cartesian_plot(df, save_path=save_path)
    
    # Show plots unless disabled
    if not args.no_show:
        plt.show()
    
    return 0

if __name__ == "__main__":
    exit(main())