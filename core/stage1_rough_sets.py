# stage1_rough_sets.py - Stage 1: Rough Set Theory
# Finds which targets are certainly valid, uncertain, or not valid.
import itertools
import numpy as np
import pandas as pd


class RoughSetTheoryEngine:

    def __init__(self, df, id_col, condition_cols, decision_col):
        self.id_col = id_col
        self.condition_cols = condition_cols
        self.decision_col = decision_col
        self.table = self.make_table(df)
        self.discretized_attributes = [
            "rcs_bin", "thermal_bin", "rf_bin",
            "mobility_bin", "depth_bin", "civ_bin",
        ]

    def make_table(self, df):
        # turn the raw numbers into simple categories (low / high etc.)
        t = df[[self.id_col, self.decision_col]].copy()
        t["rcs_bin"] = pd.cut(df["rcs_m2"], bins=[-np.inf, 0.1, 5.0, np.inf],
                              labels=["STEALTH", "MED", "LARGE"])
        t["thermal_bin"] = df["thermal_sig"].map({1: "LOW", 2: "MED", 3: "HIGH"})
        t["rf_bin"] = np.where(df["rf_emission_ghz"] > 0, "ACTIVE", "SILENT")
        t["mobility_bin"] = np.where(df["speed_kmh"] > 0, "MOBILE", "STATIC")
        t["depth_bin"] = np.where(df["est_depth_m"] > 0, "UNDERGROUND", "SURFACE")
        t["civ_bin"] = np.where(df["civilian_prox_m"] < 300, "CIV_NEAR", "CIV_FAR")
        return t

    def get_groups(self, attributes):
        # targets with identical values on these attributes go in the same group
        groups = []
        for _, rows in self.table.groupby(attributes, observed=True):
            groups.append(set(rows[self.id_col]))
        return groups

    def compute_approximations(self, target_concept=1):
        rows = self.table[self.table[self.decision_col] == target_concept]
        concept = set(rows[self.id_col])

        lower = set()
        upper = set()
        for group in self.get_groups(self.discretized_attributes):
            if group.issubset(concept):      # whole group is valid
                lower = lower | group
            if group & concept:              # group has at least one valid
                upper = upper | group

        all_ids = set(self.table[self.id_col])
        return {
            "lower_approximation": lower,
            "upper_approximation": upper,
            "boundary_region": upper - lower,
            "negative_region": all_ids - upper,
        }

    def dependency(self, attributes):
        # fraction of targets that these attributes classify with certainty
        decision_sets = []
        for _, rows in self.table.groupby(self.decision_col):
            decision_sets.append(set(rows[self.id_col]))

        certain = 0
        for group in self.get_groups(attributes):
            for d in decision_sets:
                if group.issubset(d):
                    certain += len(group)
        return certain / len(self.table)

    def compute_core_and_reducts(self):
        full = self.dependency(self.discretized_attributes)
        reducts = []

        # try smallest attribute sets first
        for size in range(1, len(self.discretized_attributes) + 1):
            for combo in itertools.combinations(self.discretized_attributes, size):
                combo = list(combo)
                if not np.isclose(self.dependency(combo), full):
                    continue
                # skip if a smaller reduct is already inside this set
                already = False
                for r in reducts:
                    if set(r).issubset(set(combo)):
                        already = True
                if not already:
                    reducts.append(combo)

        # core = attributes present in every reduct
        core = []
        if reducts:
            core = list(set(reducts[0]).intersection(*[set(r) for r in reducts]))
        return core, reducts