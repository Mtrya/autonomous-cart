#!/usr/bin/env python3
"""
Core detection algorithms for square boundary and object detection
Replicates C++ RadarHandler algorithms for testing and validation
"""

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
class Object:
    """Detected object (cart, obstacle, etc.)"""
    center_forward: float
    center_lateral: float
    radius: float
    point_count: int
    density: float = 0.0
    
    def __post_init__(self):
        self.density = self.point_count / (math.pi * self.radius**2 + 1e-6) if self.radius > 0 else 0.0

@dataclass
class SquareDetectionResult:
    """Enhanced square detection result with outlier separation"""
    square: Square
    inliers: List[RadarPoint]
    outliers: List[RadarPoint]
    interior_outliers: List[RadarPoint]
    num_inliers: int
    fitness_score: float
    edge_distances: List[float]  # [front, right, back, left]
    valid: bool

@dataclass
class ObjectDetectionResult:
    """Object detection result"""
    objects: List[Object]
    num_objects: int
    object_coverage: float
    valid: bool

@dataclass
class ComprehensiveDetectionResult:
    """Complete detection pipeline result"""
    square_result: SquareDetectionResult
    object_result: ObjectDetectionResult
    pipeline_success: bool

class SquareDetector:
    """Python implementation of the C++ square detection algorithm"""
    
    def __init__(self, inlier_threshold=100.0, max_iterations=1000):
        self.inlier_threshold = inlier_threshold
        self.max_iterations = max_iterations
        self.OFFSET = 5  # From C++ implementation
    
    def detect_square_boundary(self, points: List[RadarPoint]) -> SquareDetectionResult:
        """Main detection method - replicates C++ detectSquareBoundary"""
        if len(points) < 2:
            return SquareDetectionResult(
                Square(0, 0, 0), [], [], [], 0, 0.0, [0, 0, 0, 0], False
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
        
        # Create comprehensive result with outlier separation
        if best_inlier_count >= 8:
            outliers = [p for p in points if p not in best_inliers]
            interior_outliers = [p for p in outliers if self.is_point_inside_square(p, best_square)]
            edge_distances = self.calculate_radar_to_edge_distances(best_square)
            fitness_score = best_inlier_count / len(points)
            
            return SquareDetectionResult(
                best_square, best_inliers, outliers, interior_outliers,
                best_inlier_count, fitness_score, edge_distances, True
            )
        else:
            return SquareDetectionResult(
                Square(0, 0, 0), [], [], [], 0, 0.0, [0, 0, 0, 0], False
            )
    
    def build_square_from_two_points(self, p1: RadarPoint, p2: RadarPoint) -> Square:
        """Replicates C++ buildSquareFromTwoPoints"""
        edge_dx = p2.forward_distance - p1.forward_distance
        edge_dy = p2.lateral_distance - p1.lateral_distance
        edge_length = math.sqrt(edge_dx**2 + edge_dy**2)
        
        if edge_length < 1e-6:
            return Square(0, 0, 0)
        
        edge_dx /= edge_length
        edge_dy /= edge_length
        
        edge_mid_forward = (p1.forward_distance + p2.forward_distance) / 2.0
        edge_mid_lateral = (p1.lateral_distance + p2.lateral_distance) / 2.0
        
        perp_dx = -edge_dy
        perp_dy = edge_dx
        
        dot_product = -perp_dx * edge_mid_forward - perp_dy * edge_mid_lateral
        if dot_product < 0:
            perp_dx = -perp_dx
            perp_dy = -perp_dy
        
        center_forward = edge_mid_forward + perp_dx * Square.HALF_SIZE
        center_lateral = edge_mid_lateral + perp_dy * Square.HALF_SIZE
        orientation_deg = math.degrees(math.atan2(edge_dy, edge_dx))
        
        return Square(center_forward, center_lateral, orientation_deg)
    
    def point_to_square_edge_distance(self, point: RadarPoint, square: Square) -> float:
        """Replicates C++ pointToSquareEdgeDistance"""
        cos_theta = math.cos(math.radians(-square.orientation_deg))
        sin_theta = math.sin(math.radians(-square.orientation_deg))
        
        dx = point.forward_distance - square.center_forward
        dy = point.lateral_distance - square.center_lateral
        
        local_x = dx * cos_theta - dy * sin_theta
        local_y = dx * sin_theta + dy * cos_theta
        
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
    
    def is_point_inside_square(self, point: RadarPoint, square: Square) -> bool:
        """Check if point is inside the square"""
        cos_theta = math.cos(math.radians(-square.orientation_deg))
        sin_theta = math.sin(math.radians(-square.orientation_deg))
        
        dx = point.forward_distance - square.center_forward
        dy = point.lateral_distance - square.center_lateral
        
        local_x = dx * cos_theta - dy * sin_theta
        local_y = dx * sin_theta + dy * cos_theta
        
        return abs(local_x) <= Square.HALF_SIZE and abs(local_y) <= Square.HALF_SIZE
    
    def calculate_radar_to_edge_distances(self, square: Square) -> List[float]:
        """Replicates C++ calculateRadarToEdgeDistances"""
        cos_theta = math.cos(math.radians(-square.orientation_deg))
        sin_theta = math.sin(math.radians(-square.orientation_deg))
        
        dx = -square.center_forward
        dy = -square.center_lateral
        
        local_x = dx * cos_theta - dy * sin_theta
        local_y = dx * sin_theta + dy * cos_theta
        
        return [
            Square.HALF_SIZE - local_y,  # Front
            Square.HALF_SIZE - local_x,  # Right
            Square.HALF_SIZE + local_y,  # Back
            Square.HALF_SIZE + local_x   # Left
        ]

class ObjectDetector:
    """K-means based object detection"""
    
    def __init__(self, min_object_radius=200.0, min_points_per_object=3):
        self.min_object_radius = min_object_radius
        self.min_points_per_object = min_points_per_object
    
    def detect_objects(self, interior_outliers: List[RadarPoint], k_clusters: int = 3) -> ObjectDetectionResult:
        """K-means clustering for object detection"""
        if len(interior_outliers) < k_clusters:
            return ObjectDetectionResult([], 0, 0.0, False)
        
        try:
            from sklearn.cluster import KMeans
            points_array = np.array([[p.forward_distance, p.lateral_distance] for p in interior_outliers])
            
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(points_array)
            
            # Group points by cluster
            clusters = [[] for _ in range(k_clusters)]
            for i, label in enumerate(cluster_labels):
                clusters[label].append(interior_outliers[i])
            
            # Convert clusters to objects
            objects = []
            for cluster in clusters:
                if len(cluster) >= self.min_points_per_object:
                    obj = self.compute_object_from_cluster(cluster)
                    if obj.radius >= self.min_object_radius:
                        objects.append(obj)
            
            # Calculate coverage
            total_area = sum(math.pi * obj.radius**2 for obj in objects)
            square_area = Square.SIDE_LENGTH * Square.SIDE_LENGTH
            coverage = total_area / square_area
            
            return ObjectDetectionResult(objects, len(objects), coverage, len(objects) > 0)
            
        except ImportError:
            print("Warning: sklearn not available, using simple clustering")
            return self.simple_clustering(interior_outliers, k_clusters)
    
    def simple_clustering(self, points: List[RadarPoint], k_clusters: int) -> ObjectDetectionResult:
        """Fallback clustering without sklearn"""
        # Simple distance-based clustering
        objects = []
        used_points = set()
        
        for point in points:
            if id(point) in used_points:
                continue
                
            cluster = [point]
            used_points.add(id(point))
            
            # Find nearby points
            for other_point in points:
                if id(other_point) in used_points:
                    continue
                    
                dx = point.forward_distance - other_point.forward_distance
                dy = point.lateral_distance - other_point.lateral_distance
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 500:  # 50cm clustering radius
                    cluster.append(other_point)
                    used_points.add(id(other_point))
            
            if len(cluster) >= self.min_points_per_object:
                obj = self.compute_object_from_cluster(cluster)
                if obj.radius >= self.min_object_radius:
                    objects.append(obj)
        
        total_area = sum(math.pi * obj.radius**2 for obj in objects)
        square_area = Square.SIDE_LENGTH * Square.SIDE_LENGTH
        coverage = total_area / square_area
        
        return ObjectDetectionResult(objects, len(objects), coverage, len(objects) > 0)
    
    def compute_object_from_cluster(self, cluster: List[RadarPoint]) -> Object:
        """Compute object parameters from point cluster"""
        if not cluster:
            return Object(0, 0, 0, 0)
        
        center_x = sum(p.forward_distance for p in cluster) / len(cluster)
        center_y = sum(p.lateral_distance for p in cluster) / len(cluster)
        
        total_dist = sum(math.sqrt((p.forward_distance - center_x)**2 + (p.lateral_distance - center_y)**2) 
                        for p in cluster)
        radius = total_dist / len(cluster)
        
        return Object(center_x, center_y, radius, len(cluster))

class ComprehensiveDetector:
    """Complete detection pipeline"""
    
    def __init__(self, square_threshold=128.0, min_object_radius=20.0):
        self.square_detector = SquareDetector(inlier_threshold=square_threshold)
        self.object_detector = ObjectDetector(min_object_radius=min_object_radius)
    
    def run_detection_pipeline(self, points: List[RadarPoint], expected_objects: int = 3) -> ComprehensiveDetectionResult:
        """Run complete detection pipeline"""
        # Step 1: Square detection
        square_result = self.square_detector.detect_square_boundary(points)
        
        if not square_result.valid:
            return ComprehensiveDetectionResult(
                square_result, ObjectDetectionResult([], 0, 0.0, False), False
            )
        
        # Step 2: Object detection on interior outliers
        object_result = self.object_detector.detect_objects(square_result.interior_outliers, expected_objects)
        
        pipeline_success = square_result.valid and (object_result.valid or len(square_result.interior_outliers) < expected_objects)
        
        return ComprehensiveDetectionResult(square_result, object_result, pipeline_success)

def generate_synthetic_data(square_center_forward=500.0, square_center_lateral=200.0, 
                          orientation_deg=15.0, points_per_edge=25,
                          noise_std=50.0, outlier_ratio=0.1, 
                          missing_sectors=None, objects=None) -> List[RadarPoint]:
    """Generate comprehensive synthetic test data with radar INSIDE the square"""
    points = []
    half_size = 2000.0
    
    # Validate that radar (0,0) is inside the square
    cos_theta = math.cos(math.radians(-orientation_deg))
    sin_theta = math.sin(math.radians(-orientation_deg))
    
    # Transform radar position to square-local coordinates
    dx = 0 - square_center_forward  # Radar to square center
    dy = 0 - square_center_lateral
    local_radar_x = dx * cos_theta - dy * sin_theta
    local_radar_y = dx * sin_theta + dy * cos_theta
    
    if abs(local_radar_x) > half_size * 0.8 or abs(local_radar_y) > half_size * 0.8:
        print(f"Warning: Radar may be too close to square boundary!")
        print(f"Radar local position: ({local_radar_x:.0f}, {local_radar_y:.0f})")
        print(f"Square boundaries: ±{half_size:.0f}mm")
    
    # Generate square boundary points
    for edge in range(4):
        if missing_sectors and edge in missing_sectors:
            continue
            
        for i in range(points_per_edge):
            t = (i / max(1, points_per_edge - 1)) * 2.0 - 1.0  # [-1, 1]
            
            # Local edge coordinates (square-centered)
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
            
            # Transform to radar-centered global coordinates
            cos_theta_global = math.cos(math.radians(orientation_deg))
            sin_theta_global = math.sin(math.radians(orientation_deg))
            
            global_x = square_center_forward + local_x * cos_theta_global - local_y * sin_theta_global
            global_y = square_center_lateral + local_x * sin_theta_global + local_y * cos_theta_global
            
            # Convert to polar coordinates (from radar at origin)
            range_mm = math.sqrt(global_x**2 + global_y**2)
            bearing_deg = math.degrees(math.atan2(global_y, global_x))
            
            points.append(RadarPoint(range_mm, bearing_deg, 50))
    
    # Add objects inside square (relative to square center, then transform to global)
    if objects:
        for obj_center_local, obj_radius, obj_points in objects:
            obj_local_x, obj_local_y = obj_center_local
            
            # Validate object is inside square
            if abs(obj_local_x) > half_size - obj_radius or abs(obj_local_y) > half_size - obj_radius:
                print(f"Warning: Object at ({obj_local_x}, {obj_local_y}) may extend outside square!")
            
            for _ in range(obj_points):
                # Random point around object center (in square-local coordinates)
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, obj_radius)
                
                point_local_x = obj_local_x + radius * math.cos(angle)
                point_local_y = obj_local_y + radius * math.sin(angle)
                
                # Transform to radar-centered global coordinates
                point_global_x = square_center_forward + point_local_x * cos_theta_global - point_local_y * sin_theta_global
                point_global_y = square_center_lateral + point_local_x * sin_theta_global + point_local_y * cos_theta_global
                
                # Convert to polar
                range_mm = math.sqrt(point_global_x**2 + point_global_y**2)
                bearing_deg = math.degrees(math.atan2(point_global_y, point_global_x))
                
                points.append(RadarPoint(range_mm, bearing_deg, 40))
    
    # Add random outliers (outside square)
    num_outliers = int(len(points) * outlier_ratio)
    for _ in range(num_outliers):
        range_mm = random.uniform(500, 6000)
        bearing_deg = random.uniform(0, 360)
        points.append(RadarPoint(range_mm, bearing_deg, 20))
    
    print(f"Generated {len(points)} points: {points_per_edge*4} boundary, "
          f"{sum(len(objects) for objects in [objects] if objects) * sum(obj[2] for obj in objects) if objects else 0} object, "
          f"{num_outliers} outliers")
    
    return points