# models.py - simple data containers used by the whole project
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TargetRecord:
    target_id: str
    rcs_m2: float            # radar cross-section
    thermal_sig: int         # 1 = low, 2 = medium, 3 = high
    rf_emission_ghz: float   # radar emission (0 = silent)
    speed_kmh: float         # 0 = static
    est_depth_m: float       # how deep underground
    civilian_prox_m: float   # distance to nearest civilian building
    coord_x: float
    coord_y: float
    is_valid_military_target: int   # 1 = yes, 0 = no (used by rough sets)


@dataclass(frozen=True)
class WeaponSpec:
    weapon_name: str
    stock: int
    penetration_power: float   # 0 to 1
    blast_radius_m: float
    cost_weight: float
    speed_kmh: float


@dataclass(frozen=True)
class TargetStructureBounds:
    target_id: str
    x_bounds: tuple
    y_bounds: tuple
    z_bounds: tuple
    critical_weak_point: tuple


@dataclass
class StrikeAdvisoryEntry:
    target_id: str
    rough_set_status: str
    action: str
    predicted_class: str = None
    predicted_vulnerability_index: float = None
    enemy_air_defense_assessment: dict = None
    assigned_weapon: str = None
    strike_coordinates: dict = None
    recommended_angle: dict = None
    metrics: dict = None

    def to_dict(self):
        # keep only the fields that have a value
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result