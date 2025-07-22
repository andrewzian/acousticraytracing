from room import *
from tqdm import tqdm

class RayTracingSolution:
    def __init__(self, rays, receivers):
        self.rays = rays
        self.receivers = receivers

    def __repr__(self):
        return f"RayTracingSolution(rays={len(self.rays)}, receivers={len(self.receivers)})"
    


class RayTracingSolver:
    def __init__(self, room: Room, sources, receivers, max_time, energy_threshold=1e-6):
        self.room = room
        self.sources = sources
        self.receivers = receivers
        self.max_time = max_time
        self.energy_threshold = energy_threshold
        self.rays = []

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
                    if inter[1] > 1e-3
                ]
                # If no intersections found, break
                if not intersection_candidates:
                    break
                # Find the closest intersection and reflect the ray
                (intersection_point, _), surface = min(intersection_candidates, key=lambda x: x[0][1])
                surface.reflect(ray, intersection_point) # also records hit if receiver
        return RayTracingSolution(self.rays, self.receivers)
        

