# stage3_moga.py - Stage 3: genetic algorithm that picks a weapon for each target
import numpy as np
from core.stage2_neural_net import ThreatIntelligenceEngine


class MOGAWeaponAllocator:

    def __init__(self, targets_df, inventory, pop_size=50, generations=60):
        self.targets = targets_df.reset_index(drop=True)
        self.inventory = inventory
        self.weapon_names = list(inventory.keys())
        self.pop_size = pop_size
        self.generations = generations
        self.num_targets = len(self.targets)

    def fitness(self, chromosome):
        # chromosome = list of weapon numbers, one for each target
        # higher score = better plan
        total_kill = 0.0
        total_collateral = 0.0
        total_cost = 0.0
        used = {name: 0 for name in self.weapon_names}

        for i, w in enumerate(chromosome):
            name = self.weapon_names[w]
            weapon = self.inventory[name]
            target = self.targets.iloc[i]
            used[name] += 1

            # chance the weapon gets through enemy air defence
            threat = ThreatIntelligenceEngine.calculate_air_defense_tier(
                target["rf_emission_ghz"], target["rcs_m2"], target["speed_kmh"])
            speed_help = min(weapon.speed_kmh / 1000.0, 0.95)
            p_intercept = threat["base_interception_probability"] * (1 - speed_help)
            p_survive = 1 - p_intercept

            # chance the weapon destroys the target if it arrives
            if target["est_depth_m"] > 5.0:
                p_impact = weapon.penetration_power * 0.95
            else:
                p_impact = min(0.99, (weapon.blast_radius_m / 30.0) * 0.85)

            total_kill += p_survive * p_impact

            # penalty if civilians are inside the safety distance
            safe_distance = weapon.blast_radius_m * 3.0
            if target["civilian_prox_m"] < safe_distance:
                total_collateral += (safe_distance - target["civilian_prox_m"]) / safe_distance

            total_cost += weapon.cost_weight

        # big penalty if we use more weapons than we have in stock
        for name in self.weapon_names:
            if used[name] > self.inventory[name].stock:
                total_kill -= 10.0

        return total_kill * 10.0 - total_collateral * 8.0 - total_cost * 0.5

    def solve(self):
        n_weapons = len(self.weapon_names)

        # start with random plans
        population = []
        for _ in range(self.pop_size):
            plan = np.random.randint(0, n_weapons, size=self.num_targets).tolist()
            population.append(plan)

        best_plan = population[0]
        best_score = -np.inf

        for _ in range(self.generations):
            # score every plan and remember the best one
            for plan in population:
                score = self.fitness(plan)
                if score > best_score:
                    best_score = score
                    best_plan = plan

            # new generation: keep the best, make the rest by mixing parents
            new_population = [best_plan.copy()]
            while len(new_population) < self.pop_size:
                # pick two random parents (not a true tournament yet)
                p1 = population[np.random.choice(len(population), 3, replace=False)[0]]
                p2 = population[np.random.choice(len(population), 3, replace=False)[0]]

                # crossover: each gene comes from parent 1 or parent 2
                child = []
                for j in range(self.num_targets):
                    if np.random.rand() > 0.5:
                        child.append(p1[j])
                    else:
                        child.append(p2[j])

                # mutation: 20% chance to change one gene randomly
                if np.random.rand() < 0.20:
                    spot = np.random.randint(0, self.num_targets)
                    child[spot] = np.random.randint(0, n_weapons)

                new_population.append(child)

            population = new_population

        result = []
        for i, w in enumerate(best_plan):
            name = self.weapon_names[w]
            result.append({
                "target_id": self.targets.iloc[i]["target_id"],
                "assigned_weapon": name,
                "weapon_spec": self.inventory[name],
            })
        return result