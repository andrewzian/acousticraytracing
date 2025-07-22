from room import *
from tqdm import tqdm

class RayTracingSolution:
    def __init__(self, rays, receivers, max_time):
        self.rays = rays
        self.receivers = receivers
        self.max_time = max_time

    def __repr__(self):
        return f"RayTracingSolution(rays={len(self.rays)}, receivers={len(self.receivers)})"
    
    def aggregate_hits(self):
        """
        Aggregate hits from all receivers into a single virtual receiver
        """
        all_hits = []
        for receiver in self.receivers:
            if receiver.type != "virtual":
                print("Warning: Aggregating non-virtual receivers may lead to inaccurate results.")
            all_hits += receiver.hits
        return np.array(all_hits)
    
    def get_responses(self, kernel_func, dt=1e-3, aggregate=False):
        """
        Compute the response for each receiver using the provided kernel function
        """
        from scipy.signal import convolve
        if aggregate:
            hits_list = [self.aggregate_hits()]
        else:
            hits_list = [np.array(receiver.hits) for receiver in self.receivers]
        
        responses = []
        # Compute response for every receiver
        for hits in hits_list:
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
    
    def get_RT60(self, kernel_func=None, method="schroeder", dt=1e-3, aggregate=False):
        """
        Compute the reverberation time (RT60) for each receiver
        Valid methods: threshold, schroeder
        """
        if method not in ["threshold", "schroeder"]:
            print(f"Invalid method: {method}. Choose 'noise' or 'schroeder'.")
        RT60s = []
        
        if kernel_func:
            responses = self.get_responses(kernel_func, dt, aggregate)
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
            if aggregate:
                hits_list = [self.aggregate_hits()]
            else:
                hits_list = [np.array(receiver.hits) for receiver in self.receivers]
            if method == "threshold":
                from sklearn.linear_model import LinearRegression
                for hits in hits_list:
                    if hits.size == 0:
                        RT60s.append(None)
                        continue
                    # Fit a linear model of log10(power) on time
                    model = LinearRegression()
                    model.fit(hits[:, 0].reshape(-1, 1), np.log10(hits[:, 1]))
                    RT60s.append(-6 / model.coef_[0])
            elif method == "schroeder":
                for hits in hits_list:
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
    
    def plot(self, show_solution=False, num_rays_show=10, ray_colorscale='Viridis'):
        if show_solution and self.solution is None:
            print("No solution to show. Please run solve() first.")
            return
        import plotly.graph_objects as go

        room = self.room
        sources = self.sources
        receivers = self.solution.receivers
        fig = go.Figure()

        # Plot plane walls
        for surf in room.surfaces:
            if surf.__class__.__name__ == "PlaneWall":
                # Create a grid for the plane
                normal = np.array(surf.normal)
                point = np.array(surf.point)
                # Find two orthogonal vectors in the plane
                v = np.array([1.0, 0.0, 0.0]) if not np.allclose(normal, [1,0,0]) else np.array([0.0, 1.0, 0.0])
                v1 = np.cross(normal, v)
                v1 = v1 / np.linalg.norm(v1)
                v2 = np.cross(normal, v1)
                v2 = v2 / np.linalg.norm(v2)
                # Make a square patch
                s = max([room.bounds[dim][1] - room.bounds[dim][0] for dim in room.bounds])
                grid_x, grid_y = np.meshgrid(np.linspace(-s/2, s/2, 10), np.linspace(-s/2, s/2, 10))
                plane_points = point[:, None, None] + v1[:, None, None]*grid_x + v2[:, None, None]*grid_y
                fig.add_surface(
                    x=plane_points[0], y=plane_points[1], z=plane_points[2],
                    showscale=False, opacity=0.3, colorscale='Blues'
                )

            elif surf.__class__.__name__ == "Ellipsoid":
                center = np.array(surf.center)
                radii = np.array(surf.radii)
                u = np.linspace(0, 2 * np.pi, 30)
                v = np.linspace(0, np.pi, 15)
                x = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))
                fig.add_surface(x=x, y=y, z=z, opacity=0.5, colorscale='Reds', showscale=False)

            elif surf.__class__.__name__ == "Paraboloid":
                center = np.array(surf.center)
                a, b, c = surf.a, surf.b, surf.c
                u = np.linspace(0, 2 * np.pi, 30)
                r = np.linspace(0, surf.r_max, 15)
                U, R = np.meshgrid(u, r)
                X = center[0] + R * np.cos(U)
                Y = center[1] + R * np.sin(U)
                Z = center[2] + a * (X - center[0])**2 + b * (Y - center[1])**2 + c
                fig.add_surface(x=X, y=Y, z=Z, opacity=0.5, colorscale='Greens', showscale=False)

        # Plot receivers
        if receivers:
            for rec in receivers:
                pos = np.array(rec.position)
                radii = np.array(rec.radii)
                # Create a sphere mesh and scale by radii
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 10)
                x = pos[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
                y = pos[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
                z = pos[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))
                fig.add_surface(
                    x=x, y=y, z=z,
                    opacity=0.7, colorscale='Oranges', showscale=False,
                    name='Receiver'
                )

        # Plot sources
        if sources:
            for src in sources:
                pos = np.array(src.position)
                fig.add_trace(go.Scatter3d(
                    x=[pos[0]], y=[pos[1]], z=[pos[2]],
                    mode='markers', marker=dict(size=8, color='red', symbol='diamond'),
                    name='Source'
                ))

        if show_solution:
            # Plot rays
            indices = np.linspace(0, len(self.solution.rays)-1, num=min(num_rays_show, len(self.solution.rays)), dtype=int)
            for i in indices:
                ray = self.solution.rays[i]
                if len(ray.path) < 2:
                    continue
                path = np.array(ray.path)
                powers = np.array(ray.power_history)
                log_powers = np.log10(powers)
                # Ensure powers and path have the same length
                if len(log_powers) != len(path):
                    print("Warning: Mismatched path and power lengths.")
                    min_len = min(len(log_powers), len(path))
                    path = path[:min_len]
                    log_powers = log_powers[:min_len]
                fig.add_trace(go.Scatter3d(
                    x=path[:, 0], y=path[:, 1], z=path[:, 2],
                    mode='lines',
                    line=dict(
                        color=log_powers,
                        colorscale=ray_colorscale,
                        width=3,
                        colorbar=dict(title='Ray Log10(Power)') if ray is self.solution.rays[0] else None  # Show colorbar once
                    ),
                    opacity=0.7,
                    name='Ray Path'
                ))

        # Set room bounds
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[room.bounds['x'][0], room.bounds['x'][1]]),
                yaxis=dict(range=[room.bounds['y'][0], room.bounds['y'][1]]),
                zaxis=dict(range=[room.bounds['z'][0], room.bounds['z'][1]]),
                aspectmode='data'
            ),
            title="Interactive 3D Room Visualization"
        )
        fig.show()
            

            

