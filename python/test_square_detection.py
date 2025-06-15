"""
Square Detection Algorithm Testing and Visualization
Replicates the C++ RANSAC square detection algorithm for testing and validation
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

@dataclass
class RadarPoint:
    """Replicates C++ RadarPoint structure"""
    range_mm: float
    bearing_deg: float
    forward_distance: float
    lateral_distance: float
    quality: int
    
    def __init__(self, range_mm: float, bearing_deg: float, quality: int = 50):
        self.range_mm = range_mm
        self.bearing_deg = bearing_deg
        self.quality = quality
        
        # Convert to cartesian (same as C++ implementation)
        bearing_rad = math.radians(bearing_deg)
        self.forward_distance = range_mm * math.cos(bearing_rad)
        self.lateral_distance = range_mm * math.sin(bearing_rad)

@dataclass
class Square:
    """Replicates C++ Square structure"""
    center_forward: float
    center_lateral: float
    orientation_deg: float
    
    SIDE_LENGTH = 4000.0  # 4m in mm
    HALF_SIZE = 2000.0    # 2m in mm

@dataclass
class SquareDetectionResult:
    """Replicates C++ SquareDetectionResult structure"""
    square: Square
    inliers: List[RadarPoint]
    num_inliers: int
    fitness_score: float
    edge_distances: List[float]  # [front, right, back, left]
    valid: bool

class SquareDetector:
    """Python implementation of the C++ square detection algorithm"""
    
    def __init__(self, inlier_threshold=100.0, max_iterations=1000):
        self.inlier_threshold = inlier_threshold
        self.max_iterations = max_iterations
        self.OFFSET = 5  # From your C++ implementation
    
    def detect_square_boundary(self, points: List[RadarPoint]) -> SquareDetectionResult:
        """Main detection method - replicates C++ detectSquareBoundary"""
        if len(points) < 2:
            return SquareDetectionResult(
                Square(0, 0, 0), [], 0, 0.0, [0, 0, 0, 0], False
            )
        
        best_square = Square(0, 0, 0)
        best_inlier_count = 0
        best_inliers = []
        
        for iteration in range(self.max_iterations):
            # Sample 2 points with offset (replicates C++ strategy)
            idx1 = random.randint(0, len(points) - 1 - self.OFFSET)
            idx2 = idx1 + self.OFFSET
            
            p1 = points[idx1]
            p2 = points[idx2]
            
            # Build square from these 2 points
            candidate_square = self.build_square_from_two_points(p1, p2)
            
            # Find inliers
            inliers = self.find_square_inliers(points, candidate_square)
            
            if len(inliers) > best_inlier_count:
                best_inlier_count = len(inliers)
                best_square = candidate_square
                best_inliers = inliers
        
        # Create result
        if best_inlier_count >= 8:  # Minimum points for valid square
            edge_distances = self.calculate_radar_to_edge_distances(best_square)
            fitness_score = best_inlier_count / len(points)
            return SquareDetectionResult(
                best_square, best_inliers, best_inlier_count, 
                fitness_score, edge_distances, True
            )
        else:
            return SquareDetectionResult(
                Square(0, 0, 0), [], 0, 0.0, [0, 0, 0, 0], False
            )
    
    def build_square_from_two_points(self, p1: RadarPoint, p2: RadarPoint) -> Square:
        """Replicates C++ buildSquareFromTwoPoints"""
        # Edge direction vector (normalized)
        edge_dx = p2.forward_distance - p1.forward_distance
        edge_dy = p2.lateral_distance - p1.lateral_distance
        edge_length = math.sqrt(edge_dx**2 + edge_dy**2)
        
        if edge_length < 1e-6:
            return Square(0, 0, 0)
        
        edge_dx /= edge_length
        edge_dy /= edge_length
        
        # Midpoint of the two sample points
        edge_mid_forward = (p1.forward_distance + p2.forward_distance) / 2.0
        edge_mid_lateral = (p1.lateral_distance + p2.lateral_distance) / 2.0
        
        # Perpendicular direction pointing inward (toward radar)
        perp_dx = -edge_dy
        perp_dy = edge_dx
        
        # Check which perpendicular direction points toward radar (0,0)
        dot_product = -perp_dx * edge_mid_forward - perp_dy * edge_mid_lateral
        if dot_product < 0:
            perp_dx = -perp_dx
            perp_dy = -perp_dy
        
        # Square center is HALF_SIZE inward from edge midpoint
        center_forward = edge_mid_forward + perp_dx * Square.HALF_SIZE
        center_lateral = edge_mid_lateral + perp_dy * Square.HALF_SIZE
        
        # Orientation angle
        orientation_deg = math.degrees(math.atan2(edge_dy, edge_dx))
        
        return Square(center_forward, center_lateral, orientation_deg)
    
    def point_to_square_edge_distance(self, point: RadarPoint, square: Square) -> float:
        """Replicates C++ pointToSquareEdgeDistance"""
        # Transform to square-centered coordinate system
        cos_theta = math.cos(math.radians(-square.orientation_deg))
        sin_theta = math.sin(math.radians(-square.orientation_deg))
        
        dx = point.forward_distance - square.center_forward
        dy = point.lateral_distance - square.center_lateral
        
        local_x = dx * cos_theta - dy * sin_theta
        local_y = dx * sin_theta + dy * cos_theta
        
        # Distance to each edge
        dist_to_right = abs(local_x - Square.HALF_SIZE)
        dist_to_left = abs(local_x + Square.HALF_SIZE)
        dist_to_front = abs(local_y - Square.HALF_SIZE)
        dist_to_back = abs(local_y + Square.HALF_SIZE)
        
        return min(dist_to_right, dist_to_left, dist_to_front, dist_to_back)
    
    def find_square_inliers(self, points: List[RadarPoint], square: Square) -> List[RadarPoint]:
        """Replicates C++ findSquareInliers"""
        inliers = []
        for point in points:
            if self.point_to_square_edge_distance(point, square) <= self.inlier_threshold:
                inliers.append(point)
        return inliers
    
    def calculate_radar_to_edge_distances(self, square: Square) -> List[float]:
        """Replicates C++ calculateRadarToEdgeDistances"""
        cos_theta = math.cos(math.radians(-square.orientation_deg))
        sin_theta = math.sin(math.radians(-square.orientation_deg))
        
        # Radar to square center
        dx = -square.center_forward
        dy = -square.center_lateral
        
        local_x = dx * cos_theta - dy * sin_theta
        local_y = dx * sin_theta + dy * cos_theta
        
        # Distances to each edge: [front, right, back, left]
        return [
            Square.HALF_SIZE - local_y,  # Front
            Square.HALF_SIZE - local_x,  # Right
            Square.HALF_SIZE + local_y,  # Back
            Square.HALF_SIZE + local_x   # Left
        ]

def is_radar_inside_square(center_forward, center_lateral, orientation_deg):
    """Check if radar at (0,0) is inside the given square"""
    half_size = 2000.0
    
    # Transform radar position (0,0) to square-centered coordinate system
    cos_theta = math.cos(math.radians(-orientation_deg))
    sin_theta = math.sin(math.radians(-orientation_deg))
    
    # Radar to square center
    dx = -center_forward
    dy = -center_lateral
    
    local_x = dx * cos_theta - dy * sin_theta
    local_y = dx * sin_theta + dy * cos_theta
    
    # Check if radar is inside square bounds
    return (abs(local_x) <= half_size and abs(local_y) <= half_size)

def generate_valid_square_parameters():
    """Generate square parameters that guarantee radar is inside"""
    max_attempts = 100
    
    for _ in range(max_attempts):
        # Generate parameters with constraints to keep radar inside
        # For a 4x4m square, radar should be at most ~1.5m from center to stay well inside
        center_forward = random.uniform(1500, 5000)  # Keep square ahead of radar
        center_lateral = random.uniform(-1200, 1200)  # Limit lateral offset
        orientation_deg = random.uniform(-45, 45)     # Limit rotation
        
        if is_radar_inside_square(center_forward, center_lateral, orientation_deg):
            return center_forward, center_lateral, orientation_deg
    
    # Fallback to a guaranteed valid configuration
    print("Warning: Using fallback square parameters")
    return 3000.0, 0.0, 0.0

def generate_synthetic_square_data(center_forward=None, center_lateral=None, 
                                 orientation_deg=None, points_per_edge=25,
                                 noise_std=50.0, outlier_ratio=0.1, 
                                 missing_sectors=None) -> List[RadarPoint]:
    """Generate synthetic square boundary data with radar guaranteed inside"""
    
    # Generate valid parameters if not provided
    if center_forward is None or center_lateral is None or orientation_deg is None:
        center_forward, center_lateral, orientation_deg = generate_valid_square_parameters()
    else:
        # Validate provided parameters
        if not is_radar_inside_square(center_forward, center_lateral, orientation_deg):
            print(f"Warning: Provided parameters put radar outside square!")
            print(f"  Center: ({center_forward}, {center_lateral}), Angle: {orientation_deg}°")
            print("  Generating new valid parameters...")
            center_forward, center_lateral, orientation_deg = generate_valid_square_parameters()
    
    print(f"Square parameters: center=({center_forward:.0f}, {center_lateral:.0f}), angle={orientation_deg:.1f}°")
    
    # Verify radar is inside (debug info)
    radar_inside = is_radar_inside_square(center_forward, center_lateral, orientation_deg)
    print(f"Radar inside square: {radar_inside}")
    
    points = []
    half_size = 2000.0
    
    # Generate points on each edge
    for edge in range(4):
        for i in range(points_per_edge):
            # Skip if this edge is in missing sectors
            if missing_sectors and edge in missing_sectors:
                continue
                
            t = (i / max(1, points_per_edge - 1)) * 2.0 - 1.0  # [-1, 1]
            
            # Local edge coordinates
            if edge == 0:    # Front edge
                local_x, local_y = t * half_size, half_size
            elif edge == 1:  # Right edge
                local_x, local_y = half_size, -t * half_size
            elif edge == 2:  # Back edge
                local_x, local_y = -t * half_size, -half_size
            else:           # Left edge
                local_x, local_y = -half_size, t * half_size
            
            # Add noise
            local_x += random.gauss(0, noise_std)
            local_y += random.gauss(0, noise_std)
            
            # Transform to global coordinates
            cos_theta = math.cos(math.radians(orientation_deg))
            sin_theta = math.sin(math.radians(orientation_deg))
            
            global_x = center_forward + local_x * cos_theta - local_y * sin_theta
            global_y = center_lateral + local_x * sin_theta + local_y * cos_theta
            
            # Convert to polar
            range_mm = math.sqrt(global_x**2 + global_y**2)
            bearing_deg = math.degrees(math.atan2(global_y, global_x))
            
            points.append(RadarPoint(range_mm, bearing_deg, 50))
    
    # Add outliers (corrupted data) - but keep them realistic
    num_outliers = int(len(points) * outlier_ratio)
    for _ in range(num_outliers):
        # Generate outliers in realistic range/bearing combinations
        range_mm = random.uniform(500, 6000)
        bearing_deg = random.uniform(0, 360)
        points.append(RadarPoint(range_mm, bearing_deg, 20))
    
    return points

def run_detection_tests():
    """Run comprehensive tests of the square detection algorithm"""
    detector = SquareDetector(inlier_threshold=100.0, max_iterations=1000)
    
    test_cases = [
        {
            "name": "Perfect Square (No Noise)",
            "params": {"noise_std": 0.0, "outlier_ratio": 0.0}
        },
        {
            "name": "Light Noise",
            "params": {"noise_std": 30.0, "outlier_ratio": 0.05}
        },
        {
            "name": "Moderate Noise + Outliers", 
            "params": {"noise_std": 75.0, "outlier_ratio": 0.15}
        },
        {
            "name": "Heavy Corruption",
            "params": {"noise_std": 150.0, "outlier_ratio": 0.4}
        },
        {
            "name": "Missing Sector (Occlusion)",
            "params": {"noise_std": 50.0, "outlier_ratio": 0.1, "missing_sectors": [1]}
        },
        {
            "name": "Multiple Missing Sectors",
            "params": {"noise_std": 50.0, "outlier_ratio": 0.1, "missing_sectors": [1, 2]}
        },
        {
            "name": "Radar Near Edge (Challenging)",
            "params": {"center_forward": 2100.0, "center_lateral": -1800.0, "orientation_deg": 25.0,
                      "noise_std": 60.0, "outlier_ratio": 0.1}
        },
        {
            "name": "Tilted Square", 
            "params": {"center_forward": 4000.0, "center_lateral": 500.0, "orientation_deg": -35.0,
                      "noise_std": 40.0, "outlier_ratio": 0.08}
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {test_case['name']}")
        print(f"{'='*60}")
        
        # Generate test data
        points = generate_synthetic_square_data(**test_case["params"])
        
        # Run detection
        result = detector.detect_square_boundary(points)
        
        print(f"Generated {len(points)} points")
        print(f"Detection successful: {result.valid}")
        if result.valid:
            print(f"Inliers: {result.num_inliers} ({result.fitness_score:.1%})")
            print(f"Square center: ({result.square.center_forward:.0f}, {result.square.center_lateral:.0f})")
            print(f"Orientation: {result.square.orientation_deg:.1f}°")
            print(f"Edge distances: {[f'{d:.0f}mm' for d in result.edge_distances]}")
            
            # Verify detection makes sense
            if min(result.edge_distances) < 0:
                print("⚠️  WARNING: Negative edge distance detected!")
        
        results.append((test_case["name"], points, result))
        
        # Visualize result
        visualize_detection_result(points, result, test_case["name"])
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful_detections = sum(1 for _, _, result in results if result.valid)
    print(f"Successful detections: {successful_detections}/{len(results)}")
    
    # Analyze edge distance accuracy for successful detections
    if successful_detections > 0:
        print("\nEdge Distance Analysis:")
        for name, points, result in results:
            if result.valid:
                print(f"  {name}: edges = {[f'{d:.0f}' for d in result.edge_distances]} mm")
    
    return results

def visualize_detection_result(points: List[RadarPoint], result: SquareDetectionResult, 
                             title: str = "Square Detection Result"):
    """Visualize the detection result with matplotlib"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Left plot: Cartesian view
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{title} - Cartesian View")
    ax1.set_xlabel("Forward Distance (mm)")
    ax1.set_ylabel("Lateral Distance (mm)")
    
    # Plot all points
    for point in points:
        if point in result.inliers:
            ax1.plot(point.forward_distance, point.lateral_distance, 'go', alpha=0.6, markersize=3)
        else:
            ax1.plot(point.forward_distance, point.lateral_distance, 'r.', alpha=0.4, markersize=2)
    
    # Plot radar position
    ax1.plot(0, 0, 'b*', markersize=15, label='Radar')
    
    # Plot detected square if valid
    if result.valid:
        square = result.square
        half_size = Square.HALF_SIZE
        
        # Square corners in local coordinates
        corners = [
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size],
            [-half_size, -half_size]  # Close the square
        ]
        
        # Transform to global coordinates
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
    
    ax1.legend()
    
    # Right plot: Polar view
    ax2 = plt.subplot(122, projection='polar')
    ax2.set_title(f"{title} - Polar View")
    
    for point in points:
        bearing_rad = math.radians(point.bearing_deg)
        if point in result.inliers:
            ax2.plot(bearing_rad, point.range_mm, 'go', alpha=0.6, markersize=3)
        else:
            ax2.plot(bearing_rad, point.range_mm, 'r.', alpha=0.4, markersize=2)
    
    ax2.set_ylim(0, 7000)
    
    # Add detection info
    info_text = f"Valid: {result.valid}\n"
    if result.valid:
        info_text += f"Inliers: {result.num_inliers}\n"
        info_text += f"Fitness: {result.fitness_score:.3f}\n"
        info_text += f"Center: ({result.square.center_forward:.0f}, {result.square.center_lateral:.0f})\n"
        info_text += f"Angle: {result.square.orientation_deg:.1f}°\n"
        info_text += f"Edge distances: [{', '.join(f'{d:.0f}' for d in result.edge_distances)}]"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"/home/optics/autonomous_cart/python/plots/{title}.png")

if __name__ == "__main__":
    print("Square Detection Algorithm Testing")
    print("==================================")
    
    # Run comprehensive tests
    results = run_detection_tests()
    
    print("\nTesting complete! Check the plots for visual results.")