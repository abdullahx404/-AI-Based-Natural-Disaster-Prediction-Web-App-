"""
Clustering Module - Week 12 Requirement
Implements K-Means clustering for flood pattern analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class FloodPatternKMeans:
    """
    K-Means Clustering for flood pattern analysis.
    
    Groups flood events and weather patterns into clusters to identify:
    - Types of flood conditions (monsoon, flash flood, riverine)
    - Regional patterns
    - Seasonal patterns
    - Risk level groupings
    """
    
    def __init__(self, n_clusters: int = 5, max_iterations: int = 100, 
                 tolerance: float = 1e-4, random_state: int = None):
        """
        Initialize K-Means clustering.
        
        Args:
            n_clusters: Number of clusters (K)
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance for centroid movement
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iterations = 0
        self.cluster_sizes = None
        
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using K-Means++ algorithm.
        
        K-Means++ selects initial centroids that are well-spread out,
        leading to better convergence and results.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Choose first centroid randomly
        idx = np.random.randint(0, n_samples)
        centroids[0] = X[idx]
        
        # Choose remaining centroids with probability proportional to distance
        for k in range(1, self.n_clusters):
            # Calculate distance from each point to nearest centroid
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                min_dist = float('inf')
                for j in range(k):
                    dist = np.sum((X[i] - centroids[j]) ** 2)
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist
            
            # Choose next centroid with probability proportional to distance^2
            probabilities = distances / distances.sum()
            idx = np.random.choice(n_samples, p=probabilities)
            centroids[k] = X[idx]
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each point to nearest centroid.
        
        Returns:
            labels: Cluster assignment for each point
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            min_dist = float('inf')
            min_cluster = 0
            
            for k in range(self.n_clusters):
                dist = np.sum((X[i] - self.centroids[k]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_cluster = k
            
            labels[i] = min_cluster
        
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids to be mean of assigned points.
        
        Returns:
            new_centroids: Updated centroid positions
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # Get all points assigned to this cluster
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                new_centroids[k] = X[np.random.randint(0, len(X))]
        
        return new_centroids
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate within-cluster sum of squares (inertia).
        
        Lower inertia = tighter clusters.
        """
        inertia = 0.0
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k]) ** 2)
        
        return inertia
    
    def fit(self, X: np.ndarray) -> 'FloodPatternKMeans':
        """
        Fit K-Means clustering to data.
        
        Args:
            X: Data matrix (n_samples, n_features)
            
        Returns:
            self: Fitted model
        """
        # Initialize centroids using K-Means++
        self.centroids = self._initialize_centroids(X)
        
        print(f"\nK-Means Clustering with K={self.n_clusters}")
        print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
        print("-" * 40)
        
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            
            if (iteration + 1) % 10 == 0:
                inertia = self._calculate_inertia(X, labels)
                print(f"Iteration {iteration + 1}: Inertia = {inertia:.4f}")
            
            if centroid_shift < self.tolerance:
                print(f"Converged at iteration {iteration + 1}")
                break
        
        self.labels = labels
        self.n_iterations = iteration + 1
        self.inertia = self._calculate_inertia(X, labels)
        self.cluster_sizes = np.bincount(labels, minlength=self.n_clusters)
        
        print("-" * 40)
        print(f"Final inertia: {self.inertia:.4f}")
        print(f"Cluster sizes: {list(self.cluster_sizes)}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data matrix (n_samples, n_features)
            
        Returns:
            labels: Cluster assignments
        """
        if self.centroids is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self._assign_clusters(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return cluster labels"""
        self.fit(X)
        return self.labels
    
    def get_cluster_statistics(self, X: np.ndarray, 
                               feature_names: List[str] = None) -> Dict:
        """
        Get statistics for each cluster.
        
        Args:
            X: Original data
            feature_names: Names of features
            
        Returns:
            stats: Dictionary with cluster statistics
        """
        if self.labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        stats = {}
        
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            
            if len(cluster_points) == 0:
                continue
            
            cluster_stats = {
                'size': len(cluster_points),
                'percentage': 100 * len(cluster_points) / len(X),
                'centroid': self.centroids[k].tolist(),
                'features': {}
            }
            
            for i, name in enumerate(feature_names):
                cluster_stats['features'][name] = {
                    'mean': float(cluster_points[:, i].mean()),
                    'std': float(cluster_points[:, i].std()),
                    'min': float(cluster_points[:, i].min()),
                    'max': float(cluster_points[:, i].max())
                }
            
            stats[f'Cluster_{k}'] = cluster_stats
        
        return stats


class FloodPatternAnalyzer:
    """
    Analyzes flood patterns using K-Means clustering.
    Provides interpretable cluster labels based on weather characteristics.
    """
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = FloodPatternKMeans(n_clusters=n_clusters, random_state=42)
        self.scaler_means = None
        self.scaler_stds = None
        self.cluster_interpretations = {}
        
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance"""
        if self.scaler_means is None:
            self.scaler_means = X.mean(axis=0)
            self.scaler_stds = X.std(axis=0) + 1e-8
        
        return (X - self.scaler_means) / self.scaler_stds
    
    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        """Reverse normalization"""
        return X_norm * self.scaler_stds + self.scaler_means
    
    def fit(self, X: np.ndarray, feature_names: List[str]) -> 'FloodPatternAnalyzer':
        """
        Fit analyzer to flood data.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
        """
        self.feature_names = feature_names
        
        # Normalize data
        X_norm = self.normalize(X)
        
        # Fit K-Means
        self.kmeans.fit(X_norm)
        
        # Get cluster statistics on original scale
        centroids_orig = self.denormalize(self.kmeans.centroids)
        
        # Interpret clusters
        self._interpret_clusters(centroids_orig, feature_names)
        
        return self
    
    def _interpret_clusters(self, centroids: np.ndarray, 
                           feature_names: List[str]):
        """
        Automatically interpret clusters based on centroid values.
        """
        # Find feature indices
        prcp_idx = None
        temp_idx = None
        humid_idx = None
        
        for i, name in enumerate(feature_names):
            name_lower = name.lower()
            if 'prcp' in name_lower or 'precip' in name_lower:
                prcp_idx = i
            elif 'tavg' in name_lower or 'temp' in name_lower:
                temp_idx = i
            elif 'humid' in name_lower:
                humid_idx = i
        
        for k in range(self.n_clusters):
            centroid = centroids[k]
            
            # Determine cluster type based on characteristics
            interpretation = {
                'name': f"Pattern {k + 1}",
                'risk_level': 'Unknown',
                'description': ''
            }
            
            # Simple interpretation logic
            if prcp_idx is not None:
                prcp_value = centroid[prcp_idx]
                all_prcp = centroids[:, prcp_idx]
                prcp_percentile = (prcp_value - all_prcp.min()) / (all_prcp.max() - all_prcp.min() + 1e-8)
                
                if prcp_percentile > 0.7:
                    interpretation['name'] = "Heavy Rainfall"
                    interpretation['risk_level'] = "HIGH"
                    interpretation['description'] = "High precipitation conditions prone to flooding"
                elif prcp_percentile > 0.4:
                    interpretation['name'] = "Moderate Rain"
                    interpretation['risk_level'] = "MODERATE"
                    interpretation['description'] = "Moderate rainfall with some flood risk"
                else:
                    interpretation['name'] = "Dry Conditions"
                    interpretation['risk_level'] = "LOW"
                    interpretation['description'] = "Low precipitation, minimal flood risk"
            
            # Refine based on humidity if available
            if humid_idx is not None and prcp_idx is not None:
                humid_value = centroid[humid_idx]
                all_humid = centroids[:, humid_idx]
                humid_percentile = (humid_value - all_humid.min()) / (all_humid.max() - all_humid.min() + 1e-8)
                
                if humid_percentile > 0.7 and centroid[prcp_idx] > np.median(centroids[:, prcp_idx]):
                    interpretation['name'] = "Monsoon Pattern"
                    interpretation['description'] = "High humidity with heavy rainfall - monsoon conditions"
            
            self.cluster_interpretations[k] = interpretation
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        X_norm = self.normalize(X)
        return self.kmeans.predict(X_norm)
    
    def get_cluster_info(self, cluster_id: int) -> Dict:
        """Get interpretation for a specific cluster"""
        return self.cluster_interpretations.get(cluster_id, {})
    
    def analyze_sample(self, X_sample: np.ndarray) -> Dict:
        """
        Analyze a single sample and return cluster information.
        
        Args:
            X_sample: Single sample (1, n_features)
            
        Returns:
            analysis: Dictionary with cluster assignment and interpretation
        """
        X_sample = X_sample.reshape(1, -1)
        cluster = self.predict(X_sample)[0]
        interpretation = self.cluster_interpretations.get(cluster, {})
        
        return {
            'cluster_id': cluster,
            'pattern_name': interpretation.get('name', 'Unknown'),
            'risk_level': interpretation.get('risk_level', 'Unknown'),
            'description': interpretation.get('description', '')
        }


def find_optimal_k(X: np.ndarray, k_range: range = range(2, 11)) -> Dict:
    """
    Find optimal number of clusters using Elbow Method.
    
    Args:
        X: Data matrix
        k_range: Range of K values to try
        
    Returns:
        results: Dictionary with inertias and recommended K
    """
    print("\n" + "=" * 60)
    print("ELBOW METHOD - Finding Optimal K")
    print("=" * 60)
    
    inertias = []
    
    for k in k_range:
        kmeans = FloodPatternKMeans(n_clusters=k, random_state=42, max_iterations=50)
        kmeans.fit(X)
        inertias.append(kmeans.inertia)
        print(f"K={k}: Inertia = {kmeans.inertia:.4f}")
    
    # Find elbow using second derivative
    if len(inertias) >= 3:
        # Calculate rate of change
        diffs = np.diff(inertias)
        diff2 = np.diff(diffs)
        
        # Elbow is where second derivative is maximum (most change in slope)
        elbow_idx = np.argmax(np.abs(diff2)) + 2  # +2 because we lost 2 points from diff
        recommended_k = list(k_range)[elbow_idx] if elbow_idx < len(k_range) else k_range[len(k_range)//2]
    else:
        recommended_k = k_range[len(k_range)//2]
    
    print(f"\nRecommended K (Elbow Method): {recommended_k}")
    
    return {
        'k_values': list(k_range),
        'inertias': inertias,
        'recommended_k': recommended_k
    }


def demo_clustering():
    """Demonstrate K-Means clustering for flood patterns"""
    print("=" * 60)
    print("K-MEANS CLUSTERING FOR FLOOD PATTERN ANALYSIS - Demo")
    print("=" * 60)
    
    # Generate synthetic flood weather data
    np.random.seed(42)
    n_samples = 500
    
    # Create different weather patterns
    # Pattern 1: Heavy monsoon (high rain, high humidity)
    p1 = np.random.randn(100, 5) + np.array([25, 50, 85, 1000, 10])
    
    # Pattern 2: Flash flood (moderate temp, sudden heavy rain)
    p2 = np.random.randn(80, 5) + np.array([30, 80, 70, 995, 15])
    
    # Pattern 3: Dry season (low rain, low humidity)
    p3 = np.random.randn(150, 5) + np.array([35, 5, 40, 1015, 5])
    
    # Pattern 4: Moderate conditions
    p4 = np.random.randn(100, 5) + np.array([28, 20, 60, 1008, 8])
    
    # Pattern 5: Cold wet (low temp, moderate rain)
    p5 = np.random.randn(70, 5) + np.array([15, 30, 75, 1005, 12])
    
    X = np.vstack([p1, p2, p3, p4, p5])
    feature_names = ['Temperature', 'Precipitation', 'Humidity', 'Pressure', 'Wind Speed']
    
    print(f"\nDataset: {len(X)} weather observations")
    print(f"Features: {feature_names}")
    
    # Find optimal K
    elbow_results = find_optimal_k(X, k_range=range(2, 8))
    
    # Fit analyzer with recommended K
    print("\n" + "=" * 60)
    print(f"CLUSTERING WITH K={elbow_results['recommended_k']}")
    print("=" * 60)
    
    analyzer = FloodPatternAnalyzer(n_clusters=elbow_results['recommended_k'])
    analyzer.fit(X, feature_names)
    
    # Get cluster statistics
    print("\n--- CLUSTER INTERPRETATIONS ---")
    for k, interp in analyzer.cluster_interpretations.items():
        print(f"\nCluster {k}: {interp['name']}")
        print(f"  Risk Level: {interp['risk_level']}")
        print(f"  Description: {interp['description']}")
    
    # Analyze a sample
    print("\n--- SAMPLE ANALYSIS ---")
    test_sample = np.array([[25, 60, 80, 998, 12]])  # High rain, high humidity
    analysis = analyzer.analyze_sample(test_sample)
    print(f"Sample: {test_sample[0]}")
    print(f"Assigned Pattern: {analysis['pattern_name']}")
    print(f"Risk Level: {analysis['risk_level']}")
    
    return analyzer, elbow_results


if __name__ == "__main__":
    demo_clustering()
