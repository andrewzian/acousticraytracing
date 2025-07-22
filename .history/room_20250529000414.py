from abc import ABC, abstractmethod

class Room:
    def __init__(self, x_bounds, y_bounds, z_bounds):
        self.bounds = {
            'x': x_bounds, 
            'y': y_bounds,
            'z': z_bounds
        }
        self.surfaces = []
        self.sound_sources = []
        self.sound_receivers = []
    
    def add_surface(self, surface):
        self.surfaces.append(surface)
    
    def add_sound_source(self, sound_source):
        self.sound_sources.append(sound_source)
    
    def add_sound_receiver(self, sound_receiver):
        self.sound_receivers.append(sound_receiver)

    def trace_ray(self, ray):
        pass

class Surface:
    def __init__(self, reflection_eff=1.0):
        self.reflection_eff = reflection_eff
    
    @abstractmethod
    def intersect(self, ray):
        pass

    @abstractmethod
    def normal_at(self, point):
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
        # Implement intersection logic
        pass

    def normal_at(self, point):
        return self.normal

    def reflect(self, ray, intersection_point):
        # Implement reflection logic
        pass

class Ray:
    def __init__(self, origin, direction, energy):
        self.origin = origin
        self.position = origin
        self.direction = direction
        self.energy = energy
        self.time = 0
        self.path = []
    
    def propagate(self, distance):
        pass