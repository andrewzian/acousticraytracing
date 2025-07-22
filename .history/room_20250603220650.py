from abc import ABC, abstractmethod
import numpy as np

class Room:
    def __init__(self, bounds, dimension):
        self.dimension = dimension
        self.bounds = bounds
        self.surfaces = []
    
    def add_surface(self, surface):
        self.surfaces.append(surface)

class Surface:
    def __init__(self, reflection_eff=1.0):
        self.reflection_eff = reflection_eff
    
    @abstractmethod
    def intersect(self, ray):
        pass

    @abstractmethod
    def reflect(self, ray, intersection_point):
        pass
        
class PlaneWall(Surface):
    def __init__(self, normal, point, reflection_eff=1.0):
        super().__init__(reflection_eff)
        self.normal = normal
        self.point = point
    
    def intersect(self, ray):
        # \vec{n} \cdot ((\vec{o}+t\vec{d}) - \vec{p}) = 0
        denom = np.dot(self.normal, ray.dir)
        if np.isclose(denom, 0): # ray parallel to wall
            return None
        t = np.dot(self.normal, self.point - ray.pos) / denom
        if t < 0:
            return None
        intersection_point = ray.pos + t * ray.dir
        return intersection_point, t

    def reflect(self, ray, intersection_point):
        ray.time += np.linalg.norm(intersection_point - ray.pos) / ray.speed
        ray.pos = intersection_point
        ray.dir = ray.dir - 2 * np.dot(ray.dir, self.normal) * self.normal
        ray.dir /= np.linalg.norm(ray.dir)
        ray.path.append(intersection_point)
        ray.power = ray.power * self.reflection_eff

class Ray:
    def __init__(self, origin, direction, power, speed):
        self.origin = origin
        self.pos = origin
        self.dir = direction
        self.power = power
        self.speed = speed
        self.time = 0
        self.path = []