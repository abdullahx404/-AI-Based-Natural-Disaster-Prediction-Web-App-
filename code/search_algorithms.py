"""
Search Algorithms Module - Week 8 Requirement
Implements A*, BFS, DFS for flood evacuation route planning
"""

import heapq
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional, Union


class FloodEvacuationGrid:
    """
    Grid-based flood evacuation route finder using search algorithms.
    Models a terrain where some areas are flooded (obstacles) and 
    finds optimal evacuation routes to safe zones.
    """
    
    def __init__(self, rows: int = 20, cols: int = None):
        """
        Initialize the grid.
        
        Args:
            rows: Number of rows (or grid_size tuple)
            cols: Number of columns (optional, defaults to rows for square grid)
        """
        if isinstance(rows, tuple):
            self.rows, self.cols = rows
        else:
            self.rows = rows
            self.cols = cols if cols is not None else rows
        
        self.grid = np.zeros((self.rows, self.cols))
        self.start = None
        self.goal = None
        self.flooded_cells = set()
        self.safe_zones = []
        
    def generate_flood_scenario(self, flood_probability: float = 0.3, 
                               flood_intensity: float = None,
                               n_safe_zones: int = 2, seed: int = None):
        """
        Generate a realistic flood scenario on the grid.
        
        Args:
            flood_probability: Percentage of grid that is flooded (0-1)
            flood_intensity: Alias for flood_probability (for backward compatibility)
            n_safe_zones: Number of safe zones to create
            seed: Random seed for reproducibility
        """
        if flood_intensity is not None:
            flood_probability = flood_intensity
            
        if seed is not None:
            np.random.seed(seed)
        
        self.grid = np.zeros((self.rows, self.cols))
        self.flooded_cells = set()
        self.safe_zones = []
        
        # Create flood zones (connected regions simulating river overflow)
        num_flood_sources = max(1, int(flood_probability * 5))
        
        for _ in range(num_flood_sources):
            # Flood source point
            source_r = np.random.randint(0, self.rows)
            source_c = np.random.randint(0, self.cols)
            
            # Spread flood using BFS-like expansion
            flood_size = int(self.rows * self.cols * flood_probability / num_flood_sources)
            queue = [(source_r, source_c)]
            visited = set()
            
            while queue and len(visited) < flood_size:
                r, c = queue.pop(0)
                if (r, c) in visited:
                    continue
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    visited.add((r, c))
                    self.grid[r, c] = 2  # 2 = flooded (obstacle)
                    self.flooded_cells.add((r, c))
                    
                    # Random expansion (simulates terrain)
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if np.random.random() < 0.7:  # 70% chance to spread
                            queue.append((r + dr, c + dc))
        
        # Set safe zones
        self._set_safe_zones(n_safe_zones)
        
        # Set random start position (in non-flooded area)
        self._set_start_position()
        
        return self.grid
    
    def _set_safe_zones(self, n_safe_zones: int = 2):
        """Set safe zones (evacuation points)"""
        self.safe_zones = []
        corners = [(0, 0), (0, self.cols-1), (self.rows-1, 0), (self.rows-1, self.cols-1)]
        
        # Try corners first
        for corner in corners:
            if self.grid[corner] == 0 and len(self.safe_zones) < n_safe_zones:
                self.safe_zones.append(corner)
                
        # Add edge cells if needed
        if len(self.safe_zones) < n_safe_zones:
            edge_cells = [(0, c) for c in range(self.cols)] + \
                        [(self.rows-1, c) for c in range(self.cols)] + \
                        [(r, 0) for r in range(self.rows)] + \
                        [(r, self.cols-1) for r in range(self.rows)]
            
            np.random.shuffle(edge_cells)
            for cell in edge_cells:
                if self.grid[cell] == 0 and cell not in self.safe_zones:
                    self.safe_zones.append(cell)
                    if len(self.safe_zones) >= n_safe_zones:
                        break
        
        if not self.safe_zones:
            # Fallback: find any non-flooded cell
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid[r, c] == 0:
                        self.safe_zones.append((r, c))
                        return
    
    def _set_start_position(self):
        """Set a random start position in non-flooded area"""
        non_flooded = [(r, c) for r in range(self.rows) for c in range(self.cols) 
                       if self.grid[r, c] == 0 and (r, c) not in self.safe_zones]
        
        if non_flooded:
            self.start = non_flooded[np.random.randint(len(non_flooded))]
        else:
            self.start = (self.rows // 2, self.cols // 2)
    
    def set_start_goal(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """Manually set start and goal positions"""
        self.start = start
        self.goal = goal
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (not flooded)"""
        r, c = pos
        neighbors = []
        
        # 8-directional movement
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr, nc] == 0:  # Not flooded
                    neighbors.append((nr, nc))
        
        return neighbors
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for A*"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def euclidean_distance(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Euclidean distance"""
        return np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
    
    # ==================== A* SEARCH ====================
    def a_star_search(self, start: Tuple[int, int] = None, 
                      goal: Tuple[int, int] = None) -> Dict:
        """
        A* Search Algorithm for finding optimal evacuation route.
        
        Returns:
            Dictionary with path, cost, nodes_expanded, and success status
        """
        start = start or self.start
        goal = goal or (self.safe_zones[0] if self.safe_zones else None)
        
        if not start or not goal:
            return {"success": False, "error": "Start or goal not set"}
        
        if self.grid[start] == 1 or self.grid[goal] == 1:
            return {"success": False, "error": "Start or goal is flooded"}
        
        # Priority queue: (f_score, g_score, position, path)
        open_set = [(self.heuristic(start, goal), 0, start, [start])]
        heapq.heapify(open_set)
        
        closed_set = set()
        g_scores = {start: 0}
        nodes_expanded = 0
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            nodes_expanded += 1
            closed_set.add(current)
            
            if current == goal:
                return {
                    "success": True,
                    "algorithm": "A*",
                    "path": path,
                    "cost": g_score,
                    "nodes_expanded": nodes_expanded,
                    "path_length": len(path)
                }
            
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Cost: diagonal moves cost sqrt(2), straight moves cost 1
                move_cost = 1.414 if abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) == 2 else 1
                tentative_g = g_score + move_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, path + [neighbor]))
        
        return {"success": False, "algorithm": "A*", "error": "No path found", "nodes_expanded": nodes_expanded}
    
    # ==================== BFS ====================
    def bfs_search(self, start: Tuple[int, int] = None, 
                   goal: Tuple[int, int] = None) -> Dict:
        """
        Breadth-First Search for evacuation route.
        Finds shortest path in terms of number of steps.
        """
        start = start or self.start
        goal = goal or (self.safe_zones[0] if self.safe_zones else None)
        
        if not start or not goal:
            return {"success": False, "error": "Start or goal not set"}
        
        queue = deque([(start, [start])])
        visited = {start}
        nodes_expanded = 0
        
        while queue:
            current, path = queue.popleft()
            nodes_expanded += 1
            
            if current == goal:
                return {
                    "success": True,
                    "algorithm": "BFS",
                    "path": path,
                    "cost": len(path) - 1,
                    "nodes_expanded": nodes_expanded,
                    "path_length": len(path)
                }
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return {"success": False, "algorithm": "BFS", "error": "No path found", "nodes_expanded": nodes_expanded}
    
    # ==================== DFS ====================
    def dfs_search(self, start: Tuple[int, int] = None, 
                   goal: Tuple[int, int] = None) -> Dict:
        """
        Depth-First Search for evacuation route.
        May not find optimal path but uses less memory.
        """
        start = start or self.start
        goal = goal or (self.safe_zones[0] if self.safe_zones else None)
        
        if not start or not goal:
            return {"success": False, "error": "Start or goal not set"}
        
        stack = [(start, [start])]
        visited = set()
        nodes_expanded = 0
        
        while stack:
            current, path = stack.pop()
            
            if current in visited:
                continue
            
            visited.add(current)
            nodes_expanded += 1
            
            if current == goal:
                return {
                    "success": True,
                    "algorithm": "DFS",
                    "path": path,
                    "cost": len(path) - 1,
                    "nodes_expanded": nodes_expanded,
                    "path_length": len(path)
                }
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
        
        return {"success": False, "algorithm": "DFS", "error": "No path found", "nodes_expanded": nodes_expanded}
    
    def find_nearest_safe_zone(self, start: Tuple[int, int] = None) -> Dict:
        """Find the nearest safe zone using A* to all safe zones"""
        start = start or self.start
        best_result = None
        
        for safe_zone in self.safe_zones:
            result = self.a_star_search(start, safe_zone)
            if result["success"]:
                if best_result is None or result["cost"] < best_result["cost"]:
                    best_result = result
                    best_result["destination"] = safe_zone
        
        return best_result or {"success": False, "error": "No reachable safe zone"}
    
    def compare_algorithms(self) -> Dict:
        """Compare all search algorithms on the current scenario"""
        goal = self.safe_zones[0] if self.safe_zones else None
        
        results = {
            "A*": self.a_star_search(self.start, goal),
            "BFS": self.bfs_search(self.start, goal),
            "DFS": self.dfs_search(self.start, goal)
        }
        
        comparison = []
        for algo, result in results.items():
            comparison.append({
                "Algorithm": algo,
                "Success": result.get("success", False),
                "Path Length": result.get("path_length", "N/A"),
                "Cost": round(result.get("cost", 0), 2) if result.get("cost") else "N/A",
                "Nodes Expanded": result.get("nodes_expanded", "N/A")
            })
        
        return {"comparison": comparison, "details": results}


def demo_search_algorithms():
    """Demonstrate search algorithms"""
    print("=" * 60)
    print("FLOOD EVACUATION ROUTE PLANNING - Search Algorithms Demo")
    print("=" * 60)
    
    # Create flood scenario
    grid = FloodEvacuationGrid(grid_size=(15, 15))
    grid.generate_flood_scenario(flood_intensity=0.25, seed=42)
    
    print(f"\nGrid Size: {grid.rows}x{grid.cols}")
    print(f"Start Position: {grid.start}")
    print(f"Safe Zones: {grid.safe_zones}")
    print(f"Flooded Cells: {len(grid.flooded_cells)}")
    
    # Compare algorithms
    results = grid.compare_algorithms()
    
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON")
    print("=" * 60)
    
    for r in results["comparison"]:
        print(f"\n{r['Algorithm']}:")
        print(f"  Success: {r['Success']}")
        print(f"  Path Length: {r['Path Length']}")
        print(f"  Cost: {r['Cost']}")
        print(f"  Nodes Expanded: {r['Nodes Expanded']}")
    
    return results


if __name__ == "__main__":
    demo_search_algorithms()
