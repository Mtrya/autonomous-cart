"""
Comprehensive testing suite for square and object detection algorithms
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import random
from detection_algorithms import *

def visualize_comprehensive_result(points: List[RadarPoint], 
                                 square_result: SquareDetectionResult,
                                 object_result: ObjectDetectionResult, 
                                 title: str = "Comprehensive Detection"):
    """Enhanced visualization with objects"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Cartesian view
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{title} - Square + Objects")
    ax1.set_xlabel("Forward Distance (mm)")
    ax1.set_ylabel("Lateral Distance (mm)")
    
    # Plot points by category
    for point in points:
        if square_result.valid and point in square_result.inliers:
            ax1.plot(point.forward_distance, point.lateral_distance, 'go', alpha=0.6, markersize=3, label='Boundary' if 'Boundary' not in ([t.get_text() for t in ax1.legend().get_texts()] if ax1.legend() else True) else '')
        elif square_result.valid and point in square_result.interior_outliers:
            ax1.plot(point.forward_distance, point.lateral_distance, 'mo', alpha=0.7, markersize=4, label='Interior Objects' if 'Interior Objects' not in ([t.get_text() for t in ax1.legend().get_texts()] if ax1.legend() else True) else '')
        else:
            ax1.plot(point.forward_distance, point.lateral_distance, 'r.', alpha=0.3, markersize=2, label='Outliers' if 'Outliers' not in ([t.get_text() for t in ax1.legend().get_texts()] if ax1.legend() else True) else '')
    
    # Plot radar
    ax1.plot(0, 0, 'b*', markersize=15, label='Radar')
    
    # Plot detected square
    if square_result.valid:
        square = square_result.square
        half_size = Square.HALF_SIZE
        
        corners = [
            [-half_size, -half_size], [half_size, -half_size],
            [half_size, half_size], [-half_size, half_size], [-half_size, -half_size]
        ]
        
        cos_theta = math.cos(math.radians(square.orientation_deg))
        sin_theta = math.sin(math.radians(square.orientation_deg))
        
        global_corners = []
        for local_x, local_y in corners:
            global_x = square.center_forward + local_x * cos_theta - local_y * sin_theta
            global_y = square.center_lateral + local_x * sin_theta + local_y * cos_theta
            global_corners.append([global_x, global_y])
        
        global_corners = np.array(global_corners)
        ax1.plot(global_corners[:, 0], global_corners[:, 1], 'b-', linewidth=2, label='Detected Square')
        ax1.plot(square.center_forward, square.center_lateral, 'bo', markersize=8, label='Square Center')
    
    # Plot detected objects
    if object_result.valid:
        colors = ['red', 'orange', 'purple']
        for i, obj in enumerate(object_result.objects):
            color = colors[i % len(colors)]
            circle = plt.Circle((obj.center_forward, obj.center_lateral), 
                              obj.radius, fill=False, color=color, linewidth=2)
            ax1.add_patch(circle)
            ax1.plot(obj.center_forward, obj.center_lateral, '^', 
                    color=color, markersize=8, label=f'Object {i+1}' if i < 3 else "")
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Polar view
    ax2 = plt.subplot(122, projection='polar')
    ax2.set_title(f"{title} - Polar View")
    
    for point in points:
        bearing_rad = math.radians(point.bearing_deg)
        if square_result.valid and point in square_result.inliers:
            ax2.plot(bearing_rad, point.range_mm, 'go', alpha=0.6, markersize=3)
        elif square_result.valid and point in square_result.interior_outliers:
            ax2.plot(bearing_rad, point.range_mm, 'mo', alpha=0.7, markersize=4)
        else:
            ax2.plot(bearing_rad, point.range_mm, 'r.', alpha=0.3, markersize=2)
    
    ax2.set_ylim(0, 7000)
    
    # Info panel
    info_text = f"Square: {'✓' if square_result.valid else '✗'}\n"
    if square_result.valid:
        info_text += f"Boundary: {square_result.num_inliers} pts\n"
        info_text += f"Interior: {len(square_result.interior_outliers)} pts\n"
        info_text += f"Fitness: {square_result.fitness_score:.1%}\n"
        info_text += f"Edges: [{', '.join(f'{d:.0f}' for d in square_result.edge_distances)}]\n"
    
    info_text += f"\nObjects: {'✓' if object_result.valid else '✗'}\n"
    if object_result.valid:
        info_text += f"Count: {object_result.num_objects}\n"
        info_text += f"Coverage: {object_result.object_coverage:.1%}\n"
        for i, obj in enumerate(object_result.objects[:3]):
            info_text += f"Obj{i+1}: r={obj.radius:.0f}mm, n={obj.point_count}\n"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"plots/{title}.png")

def run_test_suite():
    """Run comprehensive test suite with radar INSIDE squares"""
    detector = ComprehensiveDetector()
    
    test_cases = [
        {
            "name": "Perfect Environment (Radar Centered)",
            "square_center": (0, 0),
            "params": {"noise_std": 0.0, "outlier_ratio": 0.0, "objects": None}
        },
        {
            "name": "Single Object - Radar Off-Center",
            "square_center": (800, -400),  # Radar at (0,0), square center offset
            "params": {
                "noise_std": 30.0, "outlier_ratio": 0.05,
                "objects": [((0, 0), 300, 15)]  # Object at square center (local coords)
            }
        },
        {
            "name": "Multiple Objects - Various Positions",
            "square_center": (600, 300),
            "params": {
                "noise_std": 50.0, "outlier_ratio": 0.1,
                "objects": [
                    ((-800, -600), 250, 12),   # Object in back-left (local coords)
                    ((600, 800), 200, 10),     # Object in front-right
                    ((0, -900), 300, 15)       # Object in back-center
                ]
            }
        },
        {
            "name": "Radar Near Edge",
            "square_center": (1500, 0),  # Radar close to left edge
            "params": {
                "noise_std": 75.0, "outlier_ratio": 0.2,
                "objects": [((400, -400), 350, 18)]  # Object in front-right area
            }
        },
        {
            "name": "Crowded Environment",
            "square_center": (400, -600),
            "params": {
                "noise_std": 40.0, "outlier_ratio": 0.1,
                "missing_sectors": [1],  # Right edge occluded
                "objects": [
                    ((-500, 500), 200, 10),    # Object 1
                    ((700, -200), 250, 12),    # Object 2  
                    ((-200, -800), 180, 8),    # Object 3
                ]
            }
        },
        {
            "name": "Heavy Corruption",
            "square_center": (200, 800),
            "params": {
                "noise_std": 100.0, "outlier_ratio": 0.4,
                "objects": [((0, 0), 400, 20)]  # Large central object
            }
        }
    ]
    
    results = []
    success_count = 0
    
    print("="*60)
    print("COMPREHENSIVE DETECTION TEST SUITE")
    print("Radar positioned INSIDE each square")
    print("="*60)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['name']}")
        print("-" * 40)
        
        square_center = test_case["square_center"]
        
        # Generate test data with radar inside square
        points = generate_synthetic_data(
            square_center_forward=square_center[0],
            square_center_lateral=square_center[1],
            orientation_deg=random.uniform(-25, 25),
            points_per_edge=20,
            **test_case["params"]
        )
        
        # Validate radar is inside generated square
        expected_square = Square(square_center[0], square_center[1], 0)
        radar_inside = abs(square_center[0]) <= 2000 and abs(square_center[1]) <= 2000
        print(f"Radar inside expected square: {'✓' if radar_inside else '✗'}")
        print(f"Square center: ({square_center[0]}, {square_center[1]})")
        
        # Run detection pipeline
        result = detector.run_detection_pipeline(points, expected_objects=3)
        
        # Analyze results
        square_success = result.square_result.valid
        object_success = result.object_result.valid or len(result.square_result.interior_outliers) < 5
        pipeline_success = result.pipeline_success
        
        if pipeline_success:
            success_count += 1
        
        print(f"Points generated: {len(points)}")
        print(f"Square detection: {'✓' if square_success else '✗'}")
        if square_success:
            sq = result.square_result.square
            print(f"  Detected center: ({sq.center_forward:.0f}, {sq.center_lateral:.0f})")
            print(f"  Boundary points: {result.square_result.num_inliers}")
            print(f"  Interior outliers: {len(result.square_result.interior_outliers)}")
            print(f"  Edge distances: [{', '.join(f'{d:.0f}' for d in result.square_result.edge_distances)}]")
        
        print(f"Object detection: {'✓' if object_success else '✗'}")
        if object_success and result.object_result.valid:
            print(f"  Objects found: {result.object_result.num_objects}")
            for j, obj in enumerate(result.object_result.objects):
                print(f"    Obj{j+1}: center=({obj.center_forward:.0f}, {obj.center_lateral:.0f}), r={obj.radius:.0f}mm")
        
        print(f"Pipeline success: {'✓' if pipeline_success else '✗'}")
        
        results.append((test_case["name"], points, result))
        
        # Visualize
        visualize_comprehensive_result(
            points, result.square_result, result.object_result, test_case["name"]
        )
    
    # Final summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Successful pipelines: {success_count}/{len(test_cases)} ({success_count/len(test_cases):.1%})")
    
    return results

if __name__ == "__main__":
    print("Square and Object Detection Algorithm Testing")
    print("=" * 50)
    
    try:
        results = run_test_suite()
        print("\n🎉 Testing complete! Check the plots for visual results.")
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()