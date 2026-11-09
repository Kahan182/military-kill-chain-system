# main.py - runs all the stages one after another (simulation, made-up data)
import json
import numpy as np

from core.models import StrikeAdvisoryEntry
from core.stage1_rough_sets import RoughSetTheoryEngine
from core.stage2_neural_net import ThreatIntelligenceEngine, TrainableTargetANN
from core.stage3_moga import MOGAWeaponAllocator
from core.stage4_pso import PSOImpactOptimizer
from core.stage5_salvo import salvo_size
from data.scenario_data import (STRUCTURAL_BOUNDS_CATALOG, WEAPONS_STOCKPILE,
                                load_targets_dataframe)


def run_pipeline(seed=42):
    np.random.seed(seed)
    df = load_targets_dataframe()

    # Stage 1: rough sets - which targets are certain, uncertain, or not valid
    rough = RoughSetTheoryEngine(
        df, "target_id",
        ["rcs_m2", "thermal_sig", "rf_emission_ghz", "speed_kmh",
         "est_depth_m", "civilian_prox_m"],
        "is_valid_military_target")
    regions = rough.compute_approximations(target_concept=1)
    core, reducts = rough.compute_core_and_reducts()
    certain = regions["lower_approximation"]
    uncertain = regions["boundary_region"]

    # Stage 3: genetic algorithm picks a weapon (only for certain targets)
    certain_df = df[df["target_id"].isin(certain)].reset_index(drop=True)
    weapon_plan = {}
    if len(certain_df) > 0:
        moga = MOGAWeaponAllocator(certain_df, WEAPONS_STOCKPILE)
        for item in moga.solve():
            weapon_plan[item["target_id"]] = item

    ann = TrainableTargetANN()      # Stage 2 network (trains when created)

    advisories = []
    for _, row in df.iterrows():
        target_id = row["target_id"]

        if target_id in certain:
            status = "LOWER (certain)"
            action = "PROCEED_TO_ASSESSMENT"
        elif target_id in uncertain:
            status = "BOUNDARY (uncertain)"
            action = "HOLD - HUMAN VERIFICATION REQUIRED"
        else:
            status = "NEGATIVE (not valid)"
            action = "NO ACTION"

        entry = StrikeAdvisoryEntry(target_id=target_id,
                                    rough_set_status=status, action=action)

        # only certain targets go through the rest of the stages
        if target_id in certain:
            metrics = {}

            # Stage 2: neural network + air defence level
            result = ann.evaluate(row)
            entry.predicted_class = result["predicted_class"]
            entry.predicted_vulnerability_index = result["vulnerability_index"]
            entry.enemy_air_defense_assessment = ThreatIntelligenceEngine.calculate_air_defense_tier(
                row["rf_emission_ghz"], row["rcs_m2"], row["speed_kmh"])

            if target_id in weapon_plan:
                weapon = weapon_plan[target_id]["weapon_spec"]
                entry.assigned_weapon = weapon_plan[target_id]["assigned_weapon"]

                # Stage 5: how many rounds are enough
                metrics["salvo"] = salvo_size(row, weapon, desired_pk=0.90)

                # moving targets: predict where they will be
                if row["speed_kmh"] > 0:
                    x, y = ann.calculate_lead_point(row["coord_x"], row["coord_y"],
                                                    row["speed_kmh"], weapon.speed_kmh)
                    metrics["predicted_lead_point"] = [x, y]

            # Stage 4: particle swarm (only if we have the structure's bounds)
            if target_id in STRUCTURAL_BOUNDS_CATALOG:
                pso = PSOImpactOptimizer(STRUCTURAL_BOUNDS_CATALOG[target_id], seed=seed)
                best = pso.optimize()
                entry.strike_coordinates = best["strike_coordinates"]
                entry.recommended_angle = best["recommended_angle"]
                metrics["pso_miss_distance"] = best["miss_distance"]

            if metrics:
                entry.metrics = metrics

        advisories.append(entry)

    return {"approximations": regions, "core": core,
            "reducts": reducts, "advisories": advisories}


if __name__ == "__main__":
    output = run_pipeline()

    print("=== STAGE 1: ROUGH SETS ===")
    for name, ids in output["approximations"].items():
        print(f"{name:20s}: {sorted(ids)}")
    print("Core   :", output["core"])
    print("Reducts:", output["reducts"])

    print("\n=== ADVISORIES ===")
    for entry in output["advisories"]:
        print(json.dumps(entry.to_dict(), indent=2, default=str))