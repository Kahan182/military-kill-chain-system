# scenario_data.py - the made-up data used by the simulation
import pandas as pd
from core.models import TargetRecord, TargetStructureBounds, WeaponSpec

# list of targets (sensor readings only)
RAW_TARGET_CATALOG = [
    TargetRecord(target_id="T_01", rcs_m2=0.15, thermal_sig=3, rf_emission_ghz=9.4,
                 speed_kmh=0.0, est_depth_m=15.0, civilian_prox_m=1200.0,
                 coord_x=104.5, coord_y=52.1, is_valid_military_target=1),

    TargetRecord(target_id="T_02", rcs_m2=18.5, thermal_sig=2, rf_emission_ghz=0.0,
                 speed_kmh=45.0, est_depth_m=0.0, civilian_prox_m=150.0,
                 coord_x=210.3, coord_y=88.4, is_valid_military_target=1),

    TargetRecord(target_id="T_03", rcs_m2=0.05, thermal_sig=1, rf_emission_ghz=0.0,
                 speed_kmh=0.0, est_depth_m=0.0, civilian_prox_m=800.0,
                 coord_x=115.0, coord_y=55.2, is_valid_military_target=1),

    TargetRecord(target_id="T_04", rcs_m2=12.0, thermal_sig=2, rf_emission_ghz=3.1,
                 speed_kmh=60.0, est_depth_m=0.5, civilian_prox_m=400.0,
                 coord_x=310.8, coord_y=12.6, is_valid_military_target=1),

    # T_05 has the same sensor readings as T_03 but is NOT a valid target (decoy)
    TargetRecord(target_id="T_05", rcs_m2=0.05, thermal_sig=1, rf_emission_ghz=0.0,
                 speed_kmh=0.0, est_depth_m=0.0, civilian_prox_m=800.0,
                 coord_x=116.2, coord_y=56.0, is_valid_military_target=0),
]

# weapons we have available
WEAPONS_STOCKPILE = {
    "W_BUNKER_BUSTER": WeaponSpec(
        weapon_name="W_BUNKER_BUSTER", stock=4, penetration_power=0.95,
        blast_radius_m=40.0, cost_weight=9.0, speed_kmh=900.0),

    "W_PRECISION_GLIDE": WeaponSpec(
        weapon_name="W_PRECISION_GLIDE", stock=12, penetration_power=0.40,
        blast_radius_m=25.0, cost_weight=4.0, speed_kmh=750.0),

    "W_LOITERING_UAV": WeaponSpec(
        weapon_name="W_LOITERING_UAV", stock=20, penetration_power=0.15,
        blast_radius_m=10.0, cost_weight=2.0, speed_kmh=180.0),
}

# 3D size of the structure and its weak point (only T_01 has one)
STRUCTURAL_BOUNDS_CATALOG = {
    "T_01": TargetStructureBounds(
        target_id="T_01",
        x_bounds=(0.0, 60.0),
        y_bounds=(0.0, 40.0),
        z_bounds=(-20.0, 0.0),
        critical_weak_point=(25.0, 18.0, -12.0),
    ),
}


def load_targets_dataframe():
    # turn the target list into a table (one row per target)
    rows = []
    for target in RAW_TARGET_CATALOG:
        rows.append(target.__dict__)
    return pd.DataFrame(rows)