"""core/stage2_threat_and_ann.py

Stage 2: Dynamic Threat Deduction & Trainable Neural Network.
1. ThreatIntelligenceEngine: Deduces enemy SAM tier from radar observables.
2. TrainableTargetANN: Learns structural vulnerability and target classification
   using He-initialization and backpropagation gradient descent (Zero hardcoded weights).
3. calculate_lead_point: Kinematic intercept predictor.
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


class ThreatIntelligenceEngine:
    """Evaluates raw radar emissions and kinematics to compute enemy air defense threat."""

    @staticmethod
    def calculate_air_defense_tier(
        rf_emission_ghz: float, rcs_m2: float, speed_kmh: float
    ) -> Dict[str, Any]:
        # 1. Fire-Control / Missile Tracking Radar (X-Band: 8.0 - 12.0 GHz)
        if 8.0 <= rf_emission_ghz <= 12.0:
            tier = 3
            threat_name = "Tier 3: High-Density IADS (Active Fire-Control Radar)"
            base_interception = 0.85

        # 2. Early Warning / Search Radar (S-Band: 2.0 - 4.0 GHz)
        elif 2.0 <= rf_emission_ghz < 4.0:
            tier = 2
            threat_name = (
                "Tier 2: Medium-Range SAM Umbrella (Acquisition Radar Active)"
            )
            base_interception = 0.55

        # 3. Large Moving Military Asset without detected active radar
        elif rcs_m2 > 10.0 and speed_kmh > 0.0:
            tier = 1
            threat_name = "Tier 1: Point Defense / Mobile SHORAD / MANPADS"
            base_interception = 0.30

        # 4. Silent / Deeply buried bunker / Small object
        else:
            tier = 0
            threat_name = "Tier 0: Minimal / No Air Defense Detected"
            base_interception = 0.05

        return {
            "sam_tier": tier,
            "threat_classification": threat_name,
            "base_interception_probability": base_interception,
        }


class TrainableTargetANN:
    """Dual-head Multi-Layer Perceptron trained dynamically via Backpropagation.

    - Head 1: Structural Vulnerability (Sigmoid Regression)
    - Head 2: Target Classification (Softmax Multi-class)
    """

    CLASS_NAMES = [
        "Hardened Underground Command Bunker",
        "Mobile Air-Defense / SAM Unit",
        "Tactical Surface Installation",
    ]

    def __init__(self, input_dim: int = 4, hidden_dim: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = 3

        # 1. DYNAMIC WEIGHT ALLOCATION (He-Normal Initialization - No Hardcoding!)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(
            2.0 / input_dim
        )
        self.b1 = np.zeros((1, hidden_dim))

        # Head 1 weights: Vulnerability regression
        self.W_vuln = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b_vuln = np.zeros((1, 1))

        # Head 2 weights: Target classification (3 classes)
        self.W_class = np.random.randn(hidden_dim, self.num_classes) * np.sqrt(
            2.0 / hidden_dim
        )
        self.b_class = np.zeros((1, self.num_classes))

        # 2. Automatically train the network on physical doctrine data upon startup
        self._auto_train_on_combat_doctrine()

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def _auto_train_on_combat_doctrine(self, epochs: int = 250, lr: float = 0.08):
        """Generates synthetic baseline combat doctrine samples and trains

        the network weights using backpropagation gradient descent.
        """
        # Doctrine features: [rcs/20, thermal/3, speed/100, depth/20]
        # Labels: [vulnerability (0.0 to 1.0)], [class_idx (0: Bunker, 1: Mobile SAM, 2: Surface)]
        training_X = np.array(
            [
                # Bunkers: high depth, static speed, small/medium RCS
                [0.01, 0.9, 0.0, 0.85],
                [0.05, 1.0, 0.0, 0.75],
                [0.02, 0.8, 0.0, 0.95],
                # Mobile SAMs: high speed, active thermal, high RCS, zero depth
                [0.85, 0.7, 0.55, 0.0],
                [0.70, 0.6, 0.45, 0.0],
                [0.90, 0.8, 0.60, 0.02],
                # Surface installations: medium RCS, static, low depth
                [0.30, 0.3, 0.0, 0.0],
                [0.40, 0.4, 0.0, 0.05],
                [0.25, 0.2, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Ground-truth vulnerability targets
        training_Y_vuln = np.array(
            [[0.88], [0.84], [0.92], [0.72], [0.75], [0.68], [0.45], [0.50], [0.40]],
            dtype=np.float64,
        )

        # Ground-truth one-hot encoded classes: [Bunker, Mobile SAM, Surface Post]
        training_Y_class = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],  # Bunkers
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],  # Mobile SAMs
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],  # Surface Posts
            ],
            dtype=np.float64,
        )

        m = training_X.shape[0]

        # Gradient Descent Backpropagation Loop
        for epoch in range(epochs):
            # --- FORWARD PASS ---
            z1 = np.dot(training_X, self.W1) + self.b1
            a1 = self._relu(z1)

            # Head 1 (Vulnerability)
            z_vuln = np.dot(a1, self.W_vuln) + self.b_vuln
            pred_vuln = self._sigmoid(z_vuln)

            # Head 2 (Classification)
            z_class = np.dot(a1, self.W_class) + self.b_class
            pred_class = self._softmax(z_class)

            # --- BACKWARD PASS (Analytical Gradients via Chain Rule) ---
            # dLoss / dZ for Sigmoid + MSE
            dZ_vuln = (pred_vuln - training_Y_vuln) * (pred_vuln * (1.0 - pred_vuln))
            dW_vuln = np.dot(a1.T, dZ_vuln) / m
            db_vuln = np.sum(dZ_vuln, axis=0, keepdims=True) / m

            # dLoss / dZ for Softmax + Cross-Entropy
            dZ_class = (pred_class - training_Y_class) / m
            dW_class = np.dot(a1.T, dZ_class)
            db_class = np.sum(dZ_class, axis=0, keepdims=True)

            # Backprop into hidden layer
            da1 = np.dot(dZ_vuln, self.W_vuln.T) + np.dot(dZ_class, self.W_class.T)
            dz1 = da1 * (z1 > 0)  # ReLU derivative
            dW1 = np.dot(training_X.T, dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m

            # --- GRADIENT DESCENT WEIGHT UPDATES ---
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W_vuln -= lr * dW_vuln
            self.b_vuln -= lr * db_vuln
            self.W_class -= lr * dW_class
            self.b_class -= lr * db_class

    def evaluate(self, row: pd.Series) -> Dict[str, Any]:
        """Runs the trained network to predict vulnerability and target class."""
        x = np.array(
            [
                [
                    min(float(row["rcs_m2"]) / 20.0, 1.0),
                    float(row["thermal_sig"]) / 3.0,
                    min(float(row["speed_kmh"]) / 100.0, 1.0),
                    min(float(row["est_depth_m"]) / 20.0, 1.0),
                ]
            ],
            dtype=np.float64,
        )

        # Forward pass through trained weights
        a1 = self._relu(np.dot(x, self.W1) + self.b1)
        pred_vuln = float(self._sigmoid(np.dot(a1, self.W_vuln) + self.b_vuln)[0, 0])
        pred_class_probs = self._softmax(np.dot(a1, self.W_class) + self.b_class)[0]

        best_class_idx = int(np.argmax(pred_class_probs))
        confidence = float(pred_class_probs[best_class_idx])

        return {
            "vulnerability_index": round(pred_vuln, 3),
            "predicted_class": self.CLASS_NAMES[best_class_idx],
            "classification_confidence": round(confidence, 2),
        }

    @staticmethod
    def calculate_lead_point(
        coord_x: float,
        coord_y: float,
        speed_kmh: float,
        weapon_speed_kmh: float,
        standoff_range_km: float = 80.0,
    ) -> Tuple[float, float]:
        """Calculates dynamic lead intercept coordinates (x^, y^)."""
        if speed_kmh <= 0.0:
            return round(coord_x, 2), round(coord_y, 2)

        flight_time_hours = standoff_range_km / weapon_speed_kmh
        displacement_km = speed_kmh * flight_time_hours

        bearing = np.pi / 4.0  # Assumed standard tactical escape bearing
        lead_x = coord_x + displacement_km * np.cos(bearing)
        lead_y = coord_y + displacement_km * np.sin(bearing)
        return round(float(lead_x), 2), round(float(lead_y), 2)