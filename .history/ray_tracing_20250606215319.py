from room import *
from tqdm import tqdm

class RayTracingSolution:
    def __init__(self, rays, receivers, max_time):
        self.rays = rays
        self.receivers = receivers
        self.max_time = max_time

    def __repr__(self):
        return f"RayTracingSolution(rays={len(self.rays)}, receivers={len(self.receivers)})"
    
    def get_responses(self, kernel_func, dt=1e-3):
        """
        Compute the response for each receiver using the provided kernel function
        """
        from scipy.signal import convolve
        responses = []
        # Compute response for every receiver
        for receiver in self.receivers:
            hits = np.array(receiver.hits)
            if hits.size == 0:
                responses.append(None)
                continue
            time_grid = np.arange(0, self.max_time, dt)
            impulse = np.zeros_like(time_grid)
            # Bin the hits into the impulse response
            indices = np.floor(hits[:, 0] / dt).astype(int)
            for idx, power in zip(indices, hits[:, 1]):
                if 0 <= idx < len(impulse):
                    impulse[idx] += power
            # Convolve with the kernel
            kernel = kernel_func(time_grid)
            kernel /= np.sum(kernel)  # Normalize kernel
            response = convolve(impulse, kernel, mode='full')[:len(impulse)]
            responses.append((time_grid, response))
        return responses
    
    def get_RT60(self, kernel_func=None, method="schroeder", dt=1e-3):
        """
        Compute the reverberation time (RT60) for each receiver
        Valid methods: threshold, schroeder
        """
        if method not in ["threshold", "schroeder"]:
            print(f"Invalid method: {method}. Choose 'noise' or 'schroeder'.")
        RT60s = []
        
        if kernel_func:
            responses = self.get_responses(kernel_func, dt)
            if method == "threshold":
                print("Warning: Using threshold method with kernel. Estimate will be inaccurate.")
                for time_grid, response in responses:
                    if response is None:
                        RT60s.append(None)
                        continue
                    peak_idx = np.argmax(response)
                    peak_val = response[peak_idx]
                    # Only consider times after the peak
                    decay_mask = response[peak_idx:] < peak_val * 1e-6
                    if np.any(decay_mask):
                        idx = peak_idx + np.argmax(decay_mask)
                        RT60s.append(time_grid[idx] - time_grid[peak_idx])
                    else:
                        RT60s.append(None)
            elif method == "schroeder":
                for time_grid, response in responses:
                    if response is None:
                        RT60s.append(None)
                        continue
                    peak_time = time_grid[np.argmax(response)]
                    # Compute cumulative energy
                    cum_energy = np.cumsum(response)
                    thresh_mask = cum_energy >= cum_energy[-1] * (1 - 1e-6)
                    if np.any(thresh_mask):
                        idx = np.argmax(thresh_mask)
                        RT60s.append(time_grid[idx] - peak_time)
                    else:
                        RT60s.append(None)
        else:
            if method == "noise":
                from sklearn.linear_model import LinearRegression
                for receiver in self.receivers:
                    hits = np.array(receiver.hits)
                    if hits.size == 0:
                        RT60s.append(None)
                        continue
                    # Fit a linear model of log10(power) on time
                    model = LinearRegression()
                    model.fit(hits[:, 0].reshape(-1, 1), np.log10(hits[:, 1]))
                    RT60s.append(-6 / model.coef_[0])
            elif method == "schroeder":
                for receiver in self.receivers:
                    hits = np.array(receiver.hits)
                    if len(hits) == 0:
                        RT60s.append(None)
                        continue
                    # Sort hits by time
                    hits = sorted(hits, key=lambda x: x[0])
                    hits = np.array(hits)
                    # Compute cumulative power
                    cum_power = np.cumsum(hits[:, 1])
                    thresh_mask = cum_power >= cum_power[-1] * (1 - 1e-6)
                    if np.any(thresh_mask):
                        idx = np.argmax(thresh_mask)
                        RT60s.append(hits[idx][0])
                    else:
                        RT60s.append(None)
        return RT60s

class RayTracingSolver:
    def __init__(self, room: Room, sources, receivers, max_time, energy_threshold=1e-6):
        self.room = room
        self.sources = sources
        self.receivers = receivers
        self.max_time = max_time
        self.energy_threshold = energy_threshold
        self.rays = []
        self.solution = None

    def initialize_rays(self):
        for source in self.sources:
            self.rays += source.initialize_rays()

    def solve(self):
        self.initialize_rays()
        # Propagate every ray
        for ray in tqdm(self.rays):
            while ray.power > self.energy_threshold and ray.time < self.max_time:
                intersection_candidates = []
                # Check for intersections with all surfaces/receivers in the room
                for surface in self.room.surfaces:
                    intersection = surface.intersect(ray)
                    if intersection:
                        intersection_candidates.append((intersection, surface))
                for receiver in self.receivers:
                    intersection = receiver.intersect(ray)
                    if intersection:
                        intersection_candidates.append((intersection, receiver))
                # Filter intersections that are out of bounds
                intersection_candidates = [
                    (inter, surf) for inter, surf in intersection_candidates
                    if all(self.room.bounds[dim][0] <= inter[0][i] <= self.room.bounds[dim][1]
                           for i, dim in enumerate(self.room.bounds))
                ]
                # Filter out self-intersections
                intersection_candidates = [
                    (inter, surf) for inter, surf in intersection_candidates
                    if inter[1] > 1e-6
                ]
                # If no intersections found, break
                if not intersection_candidates:
                    break
                # Find the closest intersection and reflect the ray
                (intersection_point, _), surface = min(intersection_candidates, key=lambda x: x[0][1])
                surface.reflect(ray, intersection_point) # also records hit if receiver
        solution = RayTracingSolution(self.rays, self.receivers, self.max_time)
        self.solution = solution
        return solution
        

