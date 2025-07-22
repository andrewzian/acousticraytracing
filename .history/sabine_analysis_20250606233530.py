from room import *
from ray_tracing import RayTracingSolution, RayTracingSolver
import pandas as pd
from tqdm import tqdm

df = pd.DataFrame(columns=["RT60", "Source", "Receiver", "Room"])

n = 1000
max_dim = 20

for _ in tqdm(range(n)):
    x_dim = np.random.uniform(0, max_dim)
    y_dim = np.random.uniform(0, max_dim)
    z_dim = np.random.uniform(0, max_dim)
    x_bounds = (-x_dim/2, x_dim/2)
    y_bounds = (-y_dim/2, y_dim/2)
    z_bounds = (-z_dim/2, z_dim/2)
    reflection_eff = np.random.uniform(0.01, 0.99)
    receiver_pos = [np.random.uniform(x_bounds[0] + 0.1, x_bounds[1] - 0.1),
                    np.random.uniform(y_bounds[0] + 0.1, y_bounds[1] - 0.1),
                    np.random.uniform(z_bounds[0] + 0.1, z_bounds[1] - 0.1)]


    volume = x_dim * y_dim * z_dim
    surface_area = 2 * (x_dim * y_dim + y_dim * z_dim + x_dim * z_dim)
    sabine_rt60 = 0.161 * volume / (surface_area * (1 - reflection_eff))
    eyring_rt60 = -0.161 * volume / (surface_area * np.log(reflection_eff))

    room = Room(bounds = {'x': x_bounds, 'y': y_bounds, 'z': z_bounds}, reflection_eff=reflection_eff, dimension=3)
    sources = [Source(position=[0, 0, 0], power=10, num_rays=int(10000 * (volume / max_dim**3)), dimension=3)]
    receivers = [Receiver(position=receiver_pos, radii=[0.1, 0.1, 0.1], type="virtual")]
    solver = RayTracingSolver(room, sources, receivers, max_time=15.0, energy_threshold=1e-10)

    solution = solver.solve()
