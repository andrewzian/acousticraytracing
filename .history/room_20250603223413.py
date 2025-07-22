from abc import ABC, abstractmethod
import numpy as np

class Room:
    def __init__(self, bounds, dimension=3):
        self.dimension = dimension
        self.bounds = bounds
        self.surfaces = []
    
    def add_surface(self, surface):
        self.surfaces.append(surface)

class Source:
    def __init__(self, position, power, num_rays, dimension=3):
        self.position = np.array(position)
        self.power = power
        self.num_rays = num_rays
    
    def initialize_rays(self):
        """
        Initialize rays uniformly from the source
        """
        rays = []
        for _ in range(self.num_rays):
            if self.dimension == 2:
                angles = np.linspace(0, 2*np.pi, self.num_rays, endpoint=False)
                for angle in angles:
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    rays.append(Ray(self.position, direction, self.power / self.num_rays, speed=1.0))
            elif self.dimension == 3:
                def fibonacci_sphere(n):
                    points = []
                    phi = np.pi * (3 - np.sqrt(5))
                    for i in range(n):
                        y = 1 - (i / float(n - 1)) * 2
                        radius = np.sqrt(1 - y * y)
                        theta = phi * i
                        x = np.cos(theta) * radius
                        z = np.sin(theta) * radius
                        points.append(np.array([x, y, z]))
                    return points
                directions = fibonacci_sphere(self.num_rays)
                for direction in directions:
                    rays.append(Ray(self.position, direction, self.power / self.num_rays, speed=1.0))
            else:
                raise ValueError(f"Unsupported dimension: {self.dimension}")
        return rays
            
    

class Receiver:
    def __init__(self, position):
        self.position = np.array(position)

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
        """
        Check if the ray intersects with the plane wall, according to
        \vec{n} \cdot ((\vec{o}+t\vec{d}) - \vec{p}) = 0
        """
        denom = np.dot(self.normal, ray.dir)
        if np.isclose(denom, 0): # ray parallel to wall
            return None
        t = np.dot(self.normal, self.point - ray.pos) / denom
        if t < 0:
            return None
        intersection_point = ray.pos + t * ray.dir
        return intersection_point, t

    def reflect(self, ray, intersection_point):
        """
        Reflect the ray off the plane wall at the intersection point
        """
        ray.time += np.linalg.norm(intersection_point - ray.pos) / ray.speed
        ray.pos = intersection_point
        ray.dir = ray.dir - 2 * np.dot(ray.dir, self.normal) * self.normal
        ray.dir /= np.linalg.norm(ray.dir)
        ray.path.append(intersection_point)
        ray.power = ray.power * self.reflection_eff

class Ray:
    def __init__(self, origin, direction, power, speed):
        self.origin = np.array(origin)
        self.pos = np.array(origin)
        self.dir = np.array(direction)
        self.dir /= np.linalg.norm(self.dir)
        self.power = power
        self.speed = speed
        self.time = 0
        self.path = []