# AI Techniques Implementation Summary

## CS351 - Artificial Intelligence Semester Project

### AI-Based Natural Disaster (Flood) Prediction Web App

---

## 📋 Project Compliance Checklist

| Week  | Requirement                 | Implementation                                          | Status |
| ----- | --------------------------- | ------------------------------------------------------- | ------ |
| 8     | Uninformed Search (BFS/DFS) | `code/search_algorithms.py` - Evacuation routing        | ✅     |
| 8     | Informed Search (A\*)       | `code/search_algorithms.py` - Optimal pathfinding       | ✅     |
| 9     | CSP                         | `code/csp_resource_allocation.py` - Resource allocation | ✅     |
| 11    | Neural Network              | `code/neural_network.py` - LSTM time-series             | ✅     |
| 12    | Clustering                  | `code/clustering.py` - K-Means pattern analysis         | ✅     |
| 12    | Reinforcement Learning      | `code/reinforcement_learning.py` - Q-Learning           | ✅     |
| Bonus | SHAP/LIME                   | `code/explainability.py` - Model interpretation         | ✅     |

---

## 🔍 Week 8: Search Algorithms

### File: `code/search_algorithms.py`

**Application**: Flood evacuation route planning

### Implemented Algorithms:

#### 1. A\* Search (Informed)

```python
class FloodEvacuationGrid:
    def a_star_search(self):
        """
        A* algorithm using Manhattan distance heuristic.
        - Optimal and complete
        - Uses priority queue (f = g + h)
        - Time: O(b^d), Space: O(b^d)
        """
```

#### 2. Breadth-First Search (Uninformed)

```python
def bfs_search(self):
    """
    BFS for shortest path in unweighted grid.
    - Optimal for unweighted graphs
    - Uses FIFO queue
    - Time: O(V+E), Space: O(V)
    """
```

#### 3. Depth-First Search (Uninformed)

```python
def dfs_search(self):
    """
    DFS for path finding (not optimal).
    - Memory efficient
    - Uses LIFO stack
    - Time: O(V+E), Space: O(V)
    """
```

### Key Features:

- Grid-based flood scenario simulation
- Safe zone identification
- Algorithm performance comparison
- Path visualization

---

## 🧩 Week 9: Constraint Satisfaction Problem

### File: `code/csp_resource_allocation.py`

**Application**: Emergency resource allocation during floods

### Problem Formulation:

- **Variables**: Evacuation shelters
- **Domains**: Available resources (medical, rescue, supplies)
- **Constraints**:
  - Each resource assigned to one location
  - Minimum requirements per shelter
  - Distance limits for deployment

### Implemented Techniques:

#### 1. AC-3 Arc Consistency

```python
def arc_consistency_3(self):
    """
    Preprocessing to reduce domains before search.
    Enforces arc consistency across all variable pairs.
    """
```

#### 2. Backtracking with Heuristics

```python
def backtracking_search(self):
    """
    Backtracking search with:
    - MRV (Minimum Remaining Values) for variable selection
    - LCV (Least Constraining Value) for value ordering
    """
```

### Key Features:

- K-Means++ initialization for smart resource placement
- Priority-based shelter allocation
- Constraint propagation for efficiency

---

## 🧬 Week 11: Neural Networks

### File: `code/neural_network.py`

**Application**: Time-series flood prediction using LSTM

### Architecture:

```
Input (sequence_length × n_features)
    ↓
LSTM Layer (64 units, tanh activation)
    ↓
Output Layer (sigmoid, flood probability)
```

### LSTM Cell Implementation:

```python
def _lstm_cell_forward(self, x_t, h_prev, c_prev):
    """
    LSTM cell with:
    - Forget gate: f_t = σ(Wf·[h_{t-1}, x_t] + bf)
    - Input gate:  i_t = σ(Wi·[h_{t-1}, x_t] + bi)
    - Cell state:  c_t = f_t * c_{t-1} + i_t * tanh(Wc·[h_{t-1}, x_t] + bc)
    - Output gate: o_t = σ(Wo·[h_{t-1}, x_t] + bo)
    - Hidden:      h_t = o_t * tanh(c_t)
    """
```

### Training Features:

- Xavier weight initialization
- Backpropagation Through Time (BPTT)
- Gradient clipping for stability
- Class imbalance handling (oversampling)

---

## 📈 Week 12: Clustering

### File: `code/clustering.py`

**Application**: Flood weather pattern analysis

### Implemented: K-Means Clustering

```python
class FloodPatternKMeans:
    """
    K-Means with:
    - K-Means++ initialization (smart centroid selection)
    - Iterative refinement until convergence
    - Inertia calculation for elbow method
    """
```

### Features:

- **Elbow Method**: Find optimal K automatically
- **Pattern Interpretation**: Auto-label clusters (Monsoon, Flash Flood, Dry)
- **Risk Assessment**: HIGH/MODERATE/LOW based on cluster characteristics

### Cluster Types Identified:

1. Heavy Rainfall (HIGH RISK)
2. Monsoon Pattern (HIGH RISK)
3. Moderate Rain (MODERATE RISK)
4. Flash Flood Conditions (HIGH RISK)
5. Dry Conditions (LOW RISK)

---

## 🎮 Week 12: Reinforcement Learning

### File: `code/reinforcement_learning.py`

**Application**: Optimal flood evacuation decisions

### Environment:

```python
class FloodEvacuationEnvironment:
    """
    States: (flood_level, population_at_risk, resources, time)
    Actions: Wait, Warn, Voluntary Evac, Mandatory Evac, Deploy Resources
    Rewards: +100/person saved, -500/casualty, -30 false alarm
    """
```

### Q-Learning Agent:

```python
class QLearningAgent:
    """
    Q-Learning with:
    - Epsilon-greedy exploration (ε decays from 1.0 to 0.01)
    - Bellman equation: Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
    - Learning rate: α = 0.1
    - Discount factor: γ = 0.95
    """
```

### Learned Policies:

- When to issue warnings
- When to begin evacuations
- Optimal resource deployment timing
- Balancing false alarms vs. missed warnings

---

## 🔬 Bonus: Explainability (SHAP & LIME)

### File: `code/explainability.py`

**Application**: Understanding model predictions

### SHAP Implementation:

```python
class SHAPExplainer:
    """
    Kernel SHAP approximation for feature importance.
    Based on Shapley values from cooperative game theory.

    Key: How much does each feature contribute to this prediction?
    """
```

### LIME Implementation:

```python
class LIMEExplainer:
    """
    Local Interpretable Model-agnostic Explanations.
    Fits simple linear model around prediction point.

    Key: What's a simple explanation for this specific case?
    """
```

### Flood-Specific Interpretation:

```python
class FloodPredictionExplainer:
    """
    Combined SHAP + LIME with flood domain knowledge.
    Generates actionable recommendations based on risk factors.
    """
```

---

## 📊 Integration in Web App

All AI modules are integrated into `app.py` with interactive Streamlit interfaces:

```python
# Navigation
st.sidebar.markdown("### 🧠 AI Techniques")
ai_page = st.sidebar.radio(
    "AI Modules",
    ["None", "🔍 Search Algorithms", "🧩 CSP Resource Allocation",
     "🧬 Neural Network (LSTM)", "📈 K-Means Clustering",
     "🎮 Reinforcement Learning", "🔬 SHAP Explainability"]
)
```

### Each Module Provides:

1. **Theory explanation** - What the technique does
2. **Interactive configuration** - Adjust parameters
3. **Visualization** - See results graphically
4. **Domain application** - How it helps flood prediction

---

## 🚀 Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py

# Or use Docker
docker-compose up --build
```

---

## 📚 References

1. Russell & Norvig - Artificial Intelligence: A Modern Approach
2. Sutton & Barto - Reinforcement Learning: An Introduction
3. Hochreiter & Schmidhuber - Long Short-Term Memory (1997)
4. Lundberg & Lee - SHAP Values (2017)
5. Ribeiro et al. - LIME: Local Interpretable Model-agnostic Explanations (2016)

---

_CS351 - Artificial Intelligence | Semester 5 Project_
