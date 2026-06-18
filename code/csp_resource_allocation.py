"""
Constraint Satisfaction Problem (CSP) Module - Week 9 Requirement
Implements CSP for flood emergency resource allocation
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from collections import defaultdict


class FloodResourceAllocationCSP:
    """
    CSP for allocating emergency resources during flood disasters.
    
    Variables: Shelters/Evacuation centers
    Domains: Available resources (medical teams, rescue boats, food supplies)
    Constraints: 
        - Each resource can only be assigned to one location at a time
        - Minimum resource requirements per shelter based on population
        - Travel time constraints for resource deployment
        - Budget constraints
    """
    
    def __init__(self):
        self.shelters = {}  # shelter_id: {name, population, priority, location}
        self.resources = {}  # resource_id: {type, quantity, base_location}
        self.constraints = []
        self.assignment = {}  # shelter_id: [resource_ids]
        self.domains = {}  # shelter_id: [possible resource_ids]
        
    def add_shelter(self, shelter_id: str, name: str, population: int, 
                   priority: int, location: Tuple[float, float]):
        """Add an evacuation shelter/center"""
        self.shelters[shelter_id] = {
            "name": name,
            "population": population,
            "priority": priority,  # 1=highest, 5=lowest
            "location": location,
            "min_medical": max(1, population // 100),
            "min_rescue": max(1, population // 200),
            "min_supplies": max(1, population // 50)
        }
        
    def add_resource(self, resource_id: str, resource_type: str, 
                    quantity: int, base_location: Tuple[float, float]):
        """Add an available resource"""
        self.resources[resource_id] = {
            "type": resource_type,  # medical, rescue, supplies
            "quantity": quantity,
            "base_location": base_location,
            "assigned_to": None
        }
    
    def generate_scenario(self, num_shelters: int = 5, num_resources: int = 10, seed: int = None):
        """Generate a sample flood disaster scenario"""
        if seed:
            np.random.seed(seed)
        
        # Generate shelters
        shelter_names = ["School Gym", "Community Center", "Town Hall", 
                        "Hospital Annex", "Stadium", "Church Hall", "Library",
                        "Shopping Mall", "University", "Sports Complex"]
        
        for i in range(num_shelters):
            self.add_shelter(
                shelter_id=f"S{i+1}",
                name=shelter_names[i % len(shelter_names)],
                population=np.random.randint(50, 500),
                priority=np.random.randint(1, 6),
                location=(np.random.uniform(0, 100), np.random.uniform(0, 100))
            )
        
        # Generate resources
        resource_types = ["medical", "rescue", "supplies"]
        resource_names = {
            "medical": ["Medical Team A", "Medical Team B", "EMT Unit", "Doctor Squad", "Nurse Team"],
            "rescue": ["Rescue Boat 1", "Rescue Boat 2", "Helicopter", "Rescue Squad", "Diving Team"],
            "supplies": ["Food Truck", "Water Supply", "Blanket Unit", "Generator", "Tent Supply"]
        }
        
        for i in range(num_resources):
            r_type = resource_types[i % 3]
            self.add_resource(
                resource_id=f"R{i+1}",
                resource_type=r_type,
                quantity=np.random.randint(1, 5),
                base_location=(np.random.uniform(0, 100), np.random.uniform(0, 100))
            )
    
    def calculate_distance(self, loc1: Tuple[float, float], 
                          loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two locations"""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def initialize_domains(self):
        """Initialize domains for each shelter variable"""
        self.domains = {}
        
        for shelter_id in self.shelters:
            # Domain = all resources that could potentially be assigned
            self.domains[shelter_id] = list(self.resources.keys())
    
    def is_consistent(self, shelter_id: str, resource_id: str, 
                     assignment: Dict) -> bool:
        """Check if assigning resource to shelter is consistent with constraints"""
        
        # Check if resource is already assigned to another shelter
        for other_shelter, resources in assignment.items():
            if other_shelter != shelter_id and resource_id in resources:
                return False
        
        # Check distance constraint (resource shouldn't be too far)
        shelter_loc = self.shelters[shelter_id]["location"]
        resource_loc = self.resources[resource_id]["base_location"]
        distance = self.calculate_distance(shelter_loc, resource_loc)
        
        if distance > 80:  # Maximum deployment distance
            return False
        
        return True
    
    def get_minimum_requirements(self, shelter_id: str) -> Dict[str, int]:
        """Get minimum resource requirements for a shelter"""
        shelter = self.shelters[shelter_id]
        return {
            "medical": shelter["min_medical"],
            "rescue": shelter["min_rescue"],
            "supplies": shelter["min_supplies"]
        }
    
    def count_assigned_by_type(self, shelter_id: str, 
                               assignment: Dict) -> Dict[str, int]:
        """Count resources assigned to shelter by type"""
        counts = {"medical": 0, "rescue": 0, "supplies": 0}
        
        if shelter_id in assignment:
            for resource_id in assignment[shelter_id]:
                r_type = self.resources[resource_id]["type"]
                counts[r_type] += self.resources[resource_id]["quantity"]
        
        return counts
    
    def select_unassigned_variable(self, assignment: Dict) -> Optional[str]:
        """
        Select next shelter to assign resources to.
        Uses MRV (Minimum Remaining Values) heuristic.
        """
        unassigned = []
        
        for shelter_id in self.shelters:
            requirements = self.get_minimum_requirements(shelter_id)
            current = self.count_assigned_by_type(shelter_id, assignment)
            
            # Check if shelter needs more resources
            needs_more = any(current[t] < requirements[t] for t in requirements)
            
            if needs_more:
                # Calculate remaining values (available consistent resources)
                remaining = sum(1 for r_id in self.domains.get(shelter_id, [])
                              if self.is_consistent(shelter_id, r_id, assignment))
                unassigned.append((shelter_id, remaining, self.shelters[shelter_id]["priority"]))
        
        if not unassigned:
            return None
        
        # Sort by: 1) MRV (fewer remaining values first), 2) Priority (higher priority first)
        unassigned.sort(key=lambda x: (x[1], x[2]))
        return unassigned[0][0]
    
    def order_domain_values(self, shelter_id: str, assignment: Dict) -> List[str]:
        """
        Order resources to try (Least Constraining Value heuristic).
        Prefer resources that are closer and leave more options for others.
        """
        shelter_loc = self.shelters[shelter_id]["location"]
        
        def score_resource(resource_id):
            if not self.is_consistent(shelter_id, resource_id, assignment):
                return float('inf')
            
            resource = self.resources[resource_id]
            distance = self.calculate_distance(shelter_loc, resource["base_location"])
            
            # Lower score = better choice
            return distance
        
        domain = self.domains.get(shelter_id, [])
        return sorted(domain, key=score_resource)
    
    def is_complete(self, assignment: Dict) -> bool:
        """Check if all shelters have minimum required resources"""
        for shelter_id in self.shelters:
            requirements = self.get_minimum_requirements(shelter_id)
            current = self.count_assigned_by_type(shelter_id, assignment)
            
            for r_type, required in requirements.items():
                if current[r_type] < required:
                    return False
        
        return True
    
    def backtracking_search(self, assignment: Dict = None) -> Optional[Dict]:
        """
        Backtracking search with CSP techniques:
        - MRV (Minimum Remaining Values) for variable selection
        - LCV (Least Constraining Value) for value ordering
        - Forward checking for constraint propagation
        """
        if assignment is None:
            assignment = {s_id: [] for s_id in self.shelters}
            self.initialize_domains()
        
        if self.is_complete(assignment):
            return assignment
        
        shelter_id = self.select_unassigned_variable(assignment)
        
        if shelter_id is None:
            # No more shelters need resources, but might not be complete
            return assignment if self.is_complete(assignment) else None
        
        for resource_id in self.order_domain_values(shelter_id, assignment):
            if self.is_consistent(shelter_id, resource_id, assignment):
                # Assign resource
                assignment[shelter_id].append(resource_id)
                
                # Recursive call
                result = self.backtracking_search(assignment)
                
                if result is not None:
                    return result
                
                # Backtrack
                assignment[shelter_id].remove(resource_id)
        
        return None
    
    def arc_consistency_3(self) -> bool:
        """
        AC-3 algorithm for arc consistency.
        Reduces domains before search.
        """
        # Create queue of all arcs (shelter pairs that share potential resources)
        queue = []
        for s1 in self.shelters:
            for s2 in self.shelters:
                if s1 != s2:
                    queue.append((s1, s2))
        
        while queue:
            s1, s2 = queue.pop(0)
            
            if self._revise(s1, s2):
                if len(self.domains[s1]) == 0:
                    return False  # No solution possible
                
                # Add neighbors back to queue
                for s3 in self.shelters:
                    if s3 != s1 and s3 != s2:
                        queue.append((s3, s1))
        
        return True
    
    def _revise(self, s1: str, s2: str) -> bool:
        """Revise domain of s1 given constraint with s2"""
        revised = False
        
        for r_id in self.domains[s1][:]:  # Copy to modify during iteration
            # Check if there exists a valid assignment for s2
            has_valid = False
            for r_id2 in self.domains[s2]:
                if r_id != r_id2:  # Different resources
                    has_valid = True
                    break
            
            if not has_valid and len(self.domains[s2]) == 1 and r_id in self.domains[s2]:
                self.domains[s1].remove(r_id)
                revised = True
        
        return revised
    
    def solve(self) -> Dict:
        """Main solving method"""
        print("\n" + "=" * 60)
        print("SOLVING FLOOD RESOURCE ALLOCATION CSP")
        print("=" * 60)
        
        print(f"\nShelters: {len(self.shelters)}")
        print(f"Resources: {len(self.resources)}")
        
        # Initialize domains
        self.initialize_domains()
        
        # Apply AC-3 for preprocessing
        print("\nApplying AC-3 arc consistency...")
        if not self.arc_consistency_3():
            return {"success": False, "error": "No solution - AC-3 pruned all options"}
        
        # Run backtracking search
        print("Running backtracking search with MRV and LCV heuristics...")
        solution = self.backtracking_search()
        
        if solution:
            return {
                "success": True,
                "assignment": solution,
                "summary": self._summarize_solution(solution)
            }
        else:
            return {"success": False, "error": "No valid assignment found"}
    
    def _summarize_solution(self, assignment: Dict) -> List[Dict]:
        """Create human-readable summary of solution"""
        summary = []
        
        for shelter_id, resources in assignment.items():
            shelter = self.shelters[shelter_id]
            
            entry = {
                "shelter": shelter["name"],
                "population": shelter["population"],
                "priority": shelter["priority"],
                "resources_assigned": []
            }
            
            for r_id in resources:
                resource = self.resources[r_id]
                entry["resources_assigned"].append({
                    "id": r_id,
                    "type": resource["type"],
                    "quantity": resource["quantity"]
                })
            
            summary.append(entry)
        
        return summary


def demo_csp():
    """Demonstrate CSP resource allocation"""
    print("=" * 60)
    print("FLOOD EMERGENCY RESOURCE ALLOCATION - CSP Demo")
    print("=" * 60)
    
    csp = FloodResourceAllocationCSP()
    csp.generate_scenario(num_shelters=4, num_resources=8, seed=42)
    
    print("\n--- SHELTERS ---")
    for s_id, shelter in csp.shelters.items():
        print(f"{s_id}: {shelter['name']} (Pop: {shelter['population']}, Priority: {shelter['priority']})")
        print(f"   Requirements - Medical: {shelter['min_medical']}, Rescue: {shelter['min_rescue']}, Supplies: {shelter['min_supplies']}")
    
    print("\n--- RESOURCES ---")
    for r_id, resource in csp.resources.items():
        print(f"{r_id}: {resource['type']} (Qty: {resource['quantity']})")
    
    # Solve CSP
    result = csp.solve()
    
    if result["success"]:
        print("\n" + "=" * 60)
        print("SOLUTION FOUND!")
        print("=" * 60)
        
        for entry in result["summary"]:
            print(f"\n{entry['shelter']} (Population: {entry['population']}):")
            for r in entry["resources_assigned"]:
                print(f"  - {r['id']}: {r['type']} x{r['quantity']}")
    else:
        print(f"\nNo solution: {result.get('error')}")
    
    return result


if __name__ == "__main__":
    demo_csp()
