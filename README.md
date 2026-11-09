# Soft Computing Tactical Strike Advisory System

A multi-stage decision-support and optimization pipeline applying soft computing and computational intelligence techniques to tactical target verification, threat evaluation, weapon allocation, trajectory optimization, and salvo planning.

---

## Architecture Overview

```
                      +-----------------------------+
                      | Raw Sensor Intelligence Feed|
                      +--------------+--------------+
                                     |
                                     v
                 [ Stage 1: Rough Set Theory (RST) Engine ]
                 * Indiscernibility partitions
                 * Lower / Boundary approximations
                 * Attribute core & reducts
                                     |
                    +----------------+----------------+
                    |                                 |
           [ Certain Targets ]              [ Uncertain Targets ]
                    |                                 |
                    v                                 v
        [ Stage 3: Genetic Algorithm ]        [ Hold for Human ]
        * Multi-objective allocation          [  Verification  ]
        * Interception & collateral risk
                    |
                    v
        [ Stage 2: Dual-Head Neural Net ]
        * Vulnerability regression
        * Multi-class target classification
        * Kinematic lead-point calculation
                    |
                    v
        [ Stage 4: Particle Swarm Optimization ]
        * 3D impact coordinate search
        * Optimal entry angle (azimuth/elevation)
                    |
                    v
        [ Stage 5: Salvo Sizing Model ]
        * Required rounds for desired P(k)
        * Stockpile & collateral assessment
                    |
                    v
        [ Actionable Strike Advisory Output ]
```

---

## Pipeline Stages

### Stage 1: Target Verification via Rough Set Theory
- Evaluates sensor observables under uncertainty and sensor noise.
- Computes lower approximations (verified military targets), upper approximations, and boundary regions (ambiguous/decoy targets requiring human verification).
- Computes attribute reducts and core features for dimensionality reduction.

### Stage 2: Dynamic Threat Intelligence & Trainable Neural Network
- **Threat Intelligence**: Evaluates RF emissions, RCS, and kinematics to deduce enemy surface-to-air missile (SAM) defense tiers (Tier 0 to Tier 3).
- **Dual-Head Multi-Layer Perceptron (MLP)**: Trained via backpropagation gradient descent using He-initialization to output:
  1. Structural vulnerability index $[0.0, 1.0]$.
  2. Multi-class target classification probabilities (Command Bunker, Mobile SAM, Surface Installation).
- **Lead Intercept Predictor**: Kinematic lead-point calculation for mobile targets based on standoff range and weapon velocity.

### Stage 3: Weapon Allocation via Multi-Objective Genetic Algorithm (MOGA)
- Optimizes target-weapon pairings over generations using elitism, tournament selection, uniform crossover, and mutation.
- Balances mission kill probability $P_k$, weapon survivability against enemy air defenses, collateral damage penalties, and inventory stockpile constraints.

### Stage 4: Precision Impact Optimization via Particle Swarm Optimization (PSO)
- Explores bounded 3D structural volumes $(x, y, z)$ alongside impact vectors $(\text{azimuth}, \text{elevation})$.
- Dynamically adapts swarm inertia ($w_{start} \to w_{end}$) and cognitive/social acceleration coefficients to converge on critical structural weak points.

### Stage 5: Munition Expenditure & Salvo Sizing
- Computes required munition counts to achieve a designated mission kill threshold (e.g., $P_k \ge 0.90$).
- Factored against weapon stockpiles and civilian proximity safety margins.

---

## Project Structure

```
sc mini_project/
├── core/
│   ├── __init__.py
│   ├── models.py            # Typed dataclasses (TargetRecord, WeaponSpec, etc.)
│   ├── stage1_rough_sets.py # Rough Set Theory engine
│   ├── stage2_neural_net.py # Threat assessment & trained neural network
│   ├── stage3_moga.py       # Multi-objective genetic algorithm allocator
│   ├── stage4_pso.py        # Particle swarm impact optimizer
│   └── stage5_salvo.py      # Salvo sizing calculation
├── data/
│   ├── __init__.py
│   └── scenario_data.py     # Scenario target catalog, stockpiles, and bounds
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Unit and integration test suite
├── .gitignore
├── requirements.txt
├── main.py                  # End-to-end simulation runner
└── README.md
```

---

## Requirements & Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Pipeline

Execute the end-to-end decision pipeline:
```bash
python main.py
```

Run test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
