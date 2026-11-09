# stage2_neural_net.py - Stage 2: threat level + neural network
import numpy as np


class ThreatIntelligenceEngine:
    # works out how dangerous the enemy air defence is from radar signals

    @staticmethod
    def calculate_air_defense_tier(rf_emission_ghz, rcs_m2, speed_kmh):
        if 8.0 <= rf_emission_ghz <= 12.0:          # fire-control radar
            tier = 3
            name = "Tier 3: High-Density IADS (Active Fire-Control Radar)"
            chance = 0.85
        elif 2.0 <= rf_emission_ghz < 4.0:          # search radar
            tier = 2
            name = "Tier 2: Medium-Range SAM Umbrella (Acquisition Radar Active)"
            chance = 0.55
        elif rcs_m2 > 10.0 and speed_kmh > 0:       # big moving target, no radar
            tier = 1
            name = "Tier 1: Point Defense / Mobile SHORAD / MANPADS"
            chance = 0.30
        else:
            tier = 0
            name = "Tier 0: Minimal / No Air Defense Detected"
            chance = 0.05

        return {
            "sam_tier": tier,
            "threat_classification": name,
            "base_interception_probability": chance,
        }


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


class TrainableTargetANN:
    # one hidden layer, two outputs:
    #   1) vulnerability (a number between 0 and 1)
    #   2) target class (bunker / mobile SAM / surface installation)

    CLASS_NAMES = [
        "Hardened Underground Command Bunker",
        "Mobile Air-Defense / SAM Unit",
        "Tactical Surface Installation",
    ]

    def __init__(self, input_dim=4, hidden_dim=8):
        # random starting weights (no hardcoded weights)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W_vuln = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b_vuln = np.zeros((1, 1))
        self.W_class = np.random.randn(hidden_dim, 3) * np.sqrt(2.0 / hidden_dim)
        self.b_class = np.zeros((1, 3))

        self.train()

    def train(self, epochs=250, lr=0.08):
        # inputs: [rcs/20, thermal/3, speed/100, depth/20]
        X = np.array([
            [0.01, 0.9, 0.0, 0.85],   # bunkers
            [0.05, 1.0, 0.0, 0.75],
            [0.02, 0.8, 0.0, 0.95],
            [0.85, 0.7, 0.55, 0.0],   # mobile SAMs
            [0.70, 0.6, 0.45, 0.0],
            [0.90, 0.8, 0.60, 0.02],
            [0.30, 0.3, 0.0, 0.0],    # surface installations
            [0.40, 0.4, 0.0, 0.05],
            [0.25, 0.2, 0.0, 0.0],
        ])
        y_vuln = np.array([[0.88], [0.84], [0.92], [0.72], [0.75],
                           [0.68], [0.45], [0.50], [0.40]])
        y_class = np.array([
            [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 0], [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
        ])
        n = X.shape[0]

        for _ in range(epochs):
            # forward pass
            z1 = X @ self.W1 + self.b1
            a1 = relu(z1)
            vuln = sigmoid(a1 @ self.W_vuln + self.b_vuln)
            probs = softmax(a1 @ self.W_class + self.b_class)

            # backward pass (error flows back from the outputs)
            d_vuln = (vuln - y_vuln) * vuln * (1 - vuln)
            d_class = (probs - y_class) / n

            dW_vuln = a1.T @ d_vuln / n
            db_vuln = d_vuln.sum(axis=0, keepdims=True) / n
            dW_class = a1.T @ d_class
            db_class = d_class.sum(axis=0, keepdims=True)

            d_hidden = d_vuln @ self.W_vuln.T + d_class @ self.W_class.T
            d_hidden = d_hidden * (z1 > 0)
            dW1 = X.T @ d_hidden / n
            db1 = d_hidden.sum(axis=0, keepdims=True) / n

            # update weights
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W_vuln -= lr * dW_vuln
            self.b_vuln -= lr * db_vuln
            self.W_class -= lr * dW_class
            self.b_class -= lr * db_class

    def evaluate(self, row):
        # scale the target's values the same way as the training data
        x = np.array([[
            min(row["rcs_m2"] / 20.0, 1.0),
            row["thermal_sig"] / 3.0,
            min(row["speed_kmh"] / 100.0, 1.0),
            min(row["est_depth_m"] / 20.0, 1.0),
        ]])

        a1 = relu(x @ self.W1 + self.b1)
        vuln = float(sigmoid(a1 @ self.W_vuln + self.b_vuln)[0, 0])
        probs = softmax(a1 @ self.W_class + self.b_class)[0]

        best = int(np.argmax(probs))
        return {
            "vulnerability_index": round(vuln, 3),
            "predicted_class": self.CLASS_NAMES[best],
            "classification_confidence": round(float(probs[best]), 2),
        }

    @staticmethod
    def calculate_lead_point(coord_x, coord_y, speed_kmh, weapon_speed_kmh,
                             standoff_range_km=80.0):
        # where a moving target will be by the time the weapon arrives
        if speed_kmh <= 0:
            return round(coord_x, 2), round(coord_y, 2)

        flight_time = standoff_range_km / weapon_speed_kmh      # hours
        distance = speed_kmh * flight_time                      # km moved
        angle = np.pi / 4                                       # assumed direction
        lead_x = coord_x + distance * np.cos(angle)
        lead_y = coord_y + distance * np.sin(angle)
        return round(float(lead_x), 2), round(float(lead_y), 2)