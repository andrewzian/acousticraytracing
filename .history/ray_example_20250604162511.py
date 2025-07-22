from room import *
from ray_tracing import RayTracingSolver

room = Room(reflection_eff=0.8, dimension=3)
sources = [Source(position=[0, 0, 0], power=10, num_rays=1000, dimension=3)]
receivers = [Receiver(position=[5, 5, 5], radii=[0.1, 0.1, 0.1], reflection_eff=0.8, type="physical")]
solver = RayTracingSolver(room, sources, receivers, max_time=10.0, energy_threshold=1e-6)

solution = solver.solve()

