from abc import ABC, abstractmethod
import numpy as np

class Surface:
    def __init__(self, reflection_eff=1.0):
        self.reflection_eff = reflection_eff
    
    @abstractmethod
    def normal_at(self, point):
        """
        Calculate the normal vector at a given point on the surface
        """
        pass
    
    @abstractmethod
    def reflect(self, ray, intersection_point):
        """
        Reflect the ray off the surface at the intersection point
        """
        ray.time += np.linalg.norm(intersection_point - ray.pos) / ray.speed
        ray.pos = intersection_point
        
        # Calculate normal at intersection point
        normal = self.normal_at(intersection_point)

        ray.dir = ray.dir - 2 * np.dot(ray.dir, normal) * normal
        if np.dot(ray.dir, normal) > 0:
            ray.dir = -ray.dir
        ray.dir /= np.linalg.norm(ray.dir)
        ray.path.append(intersection_point)
        ray.power *= self.reflection_eff

class PlaneWall(Surface):
    def __init__(self, normal, point, reflection_eff=1.0):
        super().__init__(reflection_eff)
        self.normal = np.array(normal)
        self.point = np.array(point)
    
    def intersect(self, ray):
        """
        Check if the ray intersects with the plane wall, according to
        n \cdot ((o + t * d) - p) = 0
        """
        denom = np.dot(self.normal, ray.dir)
        if np.isclose(denom, 0): # ray parallel to wall
            return None
        t = np.dot(self.normal, self.point - ray.pos) / denom
        if t <= 0:
            return None
        intersection_point = ray.pos + t * ray.dir
        return intersection_point, t

    def normal_at(self, point):
        """
        Return the normal vector of the plane wall
        """
        return self.normal / np.linalg.norm(self.normal)    

class Ellipsoid(Surface):
    def __init__(self, center, radii, reflection_eff=1.0):
        super().__init__(reflection_eff)
        self.center = np.array(center)
        self.radii = radii

    def intersect(self, ray):
        """
        Check if the ray intersects with the ellipsoid using the equation
        ((x - c_x)/a)^2 + ((y - c_y)/b)^2 + ((z - c_z)/c)^2 = 1
        """
        a, b, c = self.radii
        o = ray.pos - self.center
        d = ray.dir
        A = (d[0]**2 / a**2) + (d[1]**2 / b**2) + (d[2]**2 / c**2)
        B = 2 * ((o[0] * d[0]) / a**2 + (o[1] * d[1]) / b**2 + (o[2] * d[2]) / c**2)
        C = (o[0]**2 / a**2) + (o[1]**2 / b**2) + (o[2]**2 / c**2) - 1
        
        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            return None
        
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-B - sqrt_discriminant) / (2 * A)
        t2 = (-B + sqrt_discriminant) / (2 * A)
        
        if t1 < 0 and t2 < 0:
            return None
        
        t = min(t for t in [t1, t2] if t >= 0)
        intersection_point = ray.pos + t * ray.dir
        return intersection_point, t
    
    def normal_at(self, point):
        """
        Calculate the normal vector at a given point on the ellipsoid
        """
        normal = (point - self.center) / self.radii
        normal /= np.linalg.norm(normal)
        return normal

class Room:
    def __init__(self, bounds={'x': (-10, 10), 'y': (-10, 10), 'z': (-10, 10)}, reflection_eff=1.0, dimension=3):
        self.dimension = dimension
        self.bounds = bounds
        self.reflection_eff = reflection_eff
        self.surfaces = []
        
        # Initialize boundaries
        if dimension == 2:
            self.surfaces.append(PlaneWall(normal=[0, 1], point=[bounds['x'][0], bounds['y'][0]], reflection_eff=reflection_eff))
            self.surfaces.append(PlaneWall(normal=[0, -1], point=[bounds['x'][0], bounds['y'][1]], reflection_eff=reflection_eff))
            self.surfaces.append(PlaneWall(normal=[1, 0], point=[bounds['x'][0], bounds['y'][0]], reflection_eff=reflection_eff))
            self.surfaces.append(PlaneWall(normal=[-1, 0], point=[bounds['x'][1], bounds['y'][0]], reflection_eff=reflection_eff))
        elif dimension == 3:
            # x
            self.surfaces.append(PlaneWall([1, 0, 0], [bounds['x'][0], 0, 0], reflection_eff=reflection_eff))
            self.surfaces.append(PlaneWall([-1, 0, 0], [bounds['x'][1], 0, 0], reflection_eff=reflection_eff))
            # y
            self.surfaces.append(PlaneWall([0, 1, 0], [0, bounds['y'][0], 0], reflection_eff=reflection_eff))
            self.surfaces.append(PlaneWall([0, -1, 0], [0, bounds['y'][1], 0], reflection_eff=reflection_eff))
            # z
            self.surfaces.append(PlaneWall([0, 0, 1], [0, 0, bounds['z'][0]], reflection_eff=reflection_eff))
            self.surfaces.append(PlaneWall([0, 0, -1], [0, 0, bounds['z'][1]], reflection_eff=reflection_eff))
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")
    
    def add_surface(self, surface: Surface):
        self.surfaces.append(surface)

class Ray:
    def __init__(self, origin, direction, power, speed):
        self.origin = np.array(origin)
        self.pos = np.array(origin)
        self.dir = np.array(direction)
        self.dir /= np.linalg.norm(self.dir)
        self.power = power
        self.speed = speed
        self.time = 0
        self.path = [self.origin]

class Source:
    def __init__(self, position, power, num_rays, speed=343, dimension=3):
        self.position = np.array(position)
        self.power = power
        self.num_rays = num_rays
        self.speed = speed
        self.dimension = dimension
    
    def initialize_rays(self):
        """
        Initialize rays uniformly from the source
        """
        rays = []
        if self.dimension == 2:
            angles = np.linspace(0, 2*np.pi, self.num_rays, endpoint=False)
            for angle in angles:
                direction = np.array([np.cos(angle), np.sin(angle)])
                rays.append(Ray(self.position, direction, self.power / self.num_rays, speed=self.speed))
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
                rays.append(Ray(self.position, direction, self.power / self.num_rays, speed=self.speed))
        else:
            raise ValueError(f"Unsupported dimension: {self.dimension}")
        return rays
            
class Receiver(Ellipsoid):
    def __init__(self, position, radii=(0.1, 0.1, 0.1), reflection_eff=1.0, type="physical"):
        super().__init__(center=position, radii=radii, reflection_eff=reflection_eff)
        
        self.position = np.array(position)
        self.type = type
        self.hits = []

    def record_hit(self, ray):
        self.hits.append([ray.time, ray.power])

    def reflect(self, ray, intersection_point):
        """
        Reflect the ray off the receiver surface (if receiver type is physical)
        and record the hit
        """
        if self.type == "physical":
            super().reflect(ray, intersection_point)
        elif self.type == "virtual":
            ray.time += np.linalg.norm(intersection_point - ray.pos) / ray.speed
            ray.pos = intersection_point
            ray.path.append(intersection_point)
        else:
            raise ValueError(f"Unsupported receiver type: {self.type}")
        self.record_hit(ray)
    
