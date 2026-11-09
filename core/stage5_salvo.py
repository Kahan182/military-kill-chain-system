# stage5_salvo.py - Stage 5: how many rounds are enough for one target
import math
from core.stage2_neural_net import ThreatIntelligenceEngine


def single_shot_pk(target, weapon):
    # chance that ONE round destroys the target (same idea as the MOGA stage)
    threat = ThreatIntelligenceEngine.calculate_air_defense_tier(
        target["rf_emission_ghz"], target["rcs_m2"], target["speed_kmh"])

    # chance the round gets through enemy air defence
    speed_help = min(weapon.speed_kmh / 1000.0, 0.95)
    p_survive = 1 - threat["base_interception_probability"] * (1 - speed_help)

    # chance it destroys the target if it arrives
    if target["est_depth_m"] > 5.0:
        p_impact = weapon.penetration_power * 0.95
    else:
        p_impact = min(0.99, (weapon.blast_radius_m / 30.0) * 0.85)

    return p_survive * p_impact


def salvo_size(target, weapon, desired_pk=0.90):
    # formula: 1 - (1 - p)^n >= desired_pk   (rounds assumed independent)
    p = single_shot_pk(target, weapon)
    needed = math.ceil(math.log(1 - desired_pk) / math.log(1 - p))
    used = min(needed, weapon.stock)       # can't use more than we have

    # collateral risk for each round fired
    safe_distance = weapon.blast_radius_m * 3.0
    risk_per_round = max(0.0, (safe_distance - target["civilian_prox_m"]) / safe_distance)

    return {
        "single_shot_pk": round(p, 3),
        "rounds_needed": needed,
        "rounds_available": weapon.stock,
        "achieved_pk": round(1 - (1 - p) ** used, 3),
        "stock_limited": needed > weapon.stock,
        "collateral_risk_total": round(risk_per_round * used, 3),
    }