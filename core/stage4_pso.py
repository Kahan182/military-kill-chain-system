# stage4_pso.py - Stage 4: Particle Swarm Optimization
# A swarm of "particles" moves around inside the structure's bounds looking
# for the best point. Each particle is [x, y, z, azimuth, elevation].
import numpy as np


class PSOImpactOptimizer:

    def __init__(self, bounds, n_particles=30, iterations=80,
                 w_start=0.9, w_end=0.4, c1=1.8, c2=1.8, seed=None):
        self.n_particles = n_particles
        self.iterations = iterations
        self.w_start = w_start      # inertia at the start (explore)
        self.w_end = w_end          # inertia at the end (settle down)
        self.c1 = c1                # pull towards the particle's own best
        self.c2 = c2                # pull towards the swarm's best
        self.rng = np.random.default_rng(seed)

        # search limits for [x, y, z, azimuth, elevation]
        self.low = np.array([bounds.x_bounds[0], bounds.y_bounds[0],
                             bounds.z_bounds[0], 0.0, 30.0])
        self.high = np.array([bounds.x_bounds[1], bounds.y_bounds[1],
                              bounds.z_bounds[1], 360.0, 90.0])
        self.weak_point = np.array(bounds.critical_weak_point, dtype=float)

    def fitness(self, p):
        # lower is better: distance to the weak point + small angle penalty
        distance = np.linalg.norm(p[:3] - self.weak_point)
        angle_penalty = 0.05 * (90.0 - p[4])
        return float(distance + angle_penalty)

    def optimize(self):
        dims = len(self.low)
        span = self.high - self.low
        max_speed = 0.2 * span

        # random start positions and speeds
        pos = self.low + self.rng.random((self.n_particles, dims)) * span
        vel = (self.rng.random((self.n_particles, dims)) - 0.5) * span * 0.1

        # each particle's best so far, and the swarm's best so far
        own_best = pos.copy()
        own_best_score = np.array([self.fitness(p) for p in pos])
        i = int(np.argmin(own_best_score))
        swarm_best = own_best[i].copy()
        swarm_best_score = float(own_best_score[i])
        history = [swarm_best_score]

        for step in range(self.iterations):
            # inertia goes from w_start down to w_end
            w = self.w_start - (self.w_start - self.w_end) * step / (self.iterations - 1)

            r1 = self.rng.random((self.n_particles, dims))
            r2 = self.rng.random((self.n_particles, dims))

            # new speed = keep some old speed + pull to own best + pull to swarm best
            vel = (w * vel
                   + self.c1 * r1 * (own_best - pos)
                   + self.c2 * r2 * (swarm_best - pos))
            vel = np.clip(vel, -max_speed, max_speed)

            # move, and stay inside the limits
            pos = np.clip(pos + vel, self.low, self.high)

            # update personal bests
            scores = np.array([self.fitness(p) for p in pos])
            improved = scores < own_best_score
            own_best[improved] = pos[improved]
            own_best_score[improved] = scores[improved]

            # update swarm best
            if own_best_score.min() < swarm_best_score:
                i = int(np.argmin(own_best_score))
                swarm_best = own_best[i].copy()
                swarm_best_score = float(own_best_score[i])

            history.append(swarm_best_score)

        miss = np.linalg.norm(swarm_best[:3] - self.weak_point)
        return {
            "strike_coordinates": {
                "x": round(float(swarm_best[0]), 2),
                "y": round(float(swarm_best[1]), 2),
                "z": round(float(swarm_best[2]), 2),
            },
            "recommended_angle": {
                "azimuth_deg": round(float(swarm_best[3]), 1),
                "elevation_deg": round(float(swarm_best[4]), 1),
            },
            "best_fitness": round(swarm_best_score, 3),
            "miss_distance": round(float(miss), 3),
            "convergence_history": history,
        }