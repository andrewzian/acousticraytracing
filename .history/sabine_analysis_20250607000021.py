from room import *
from ray_tracing import RayTracingSolution, RayTracingSolver
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def gaussian(mu=0, sigma=0.1):
    return lambda x: np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

df = pd.DataFrame(columns=["x_dim", "y_dim", "z_dim", "reflection_eff", 
                           "receiver_x", "receiver_y", "receiver_z", 
                           "volume", "surface_area", 
                           "sabine_rt60", "eyring_rt60", 
                           "rt60_threshold", "rt60_schroeder", "rt60_gaussian_schroeder"])

n = 10
max_dim = 20
max_rays = 10000
reflection_eff_bounds = (0.20, 0.95)

for _ in tqdm(range(n)):
    x_dim = np.random.uniform(0, max_dim)
    y_dim = np.random.uniform(0, max_dim)
    z_dim = np.random.uniform(0, max_dim)
    x_bounds = (-x_dim/2, x_dim/2)
    y_bounds = (-y_dim/2, y_dim/2)
    z_bounds = (-z_dim/2, z_dim/2)
    reflection_eff = np.random.uniform(*reflection_eff_bounds)
    receiver_pos = [np.random.uniform(x_bounds[0] + 0.1, x_bounds[1] - 0.1),
                    np.random.uniform(y_bounds[0] + 0.1, y_bounds[1] - 0.1),
                    np.random.uniform(z_bounds[0] + 0.1, z_bounds[1] - 0.1)]

    volume = x_dim * y_dim * z_dim
    surface_area = 2 * (x_dim * y_dim + y_dim * z_dim + x_dim * z_dim)
    sabine_rt60 = 0.161 * volume / (surface_area * (1 - reflection_eff))
    eyring_rt60 = -0.161 * volume / (surface_area * np.log(reflection_eff))

    room = Room(bounds = {'x': x_bounds, 'y': y_bounds, 'z': z_bounds}, reflection_eff=reflection_eff, dimension=3)
    
    power = 10
    num_rays = int(max_rays * (volume / max_dim**3))
    sources = [Source(position=[0, 0, 0], power=power, num_rays=num_rays, dimension=3)]
    receivers = [Receiver(position=receiver_pos, radii=[0.1, 0.1, 0.1], type="virtual")]
    solver = RayTracingSolver(room, sources, receivers, max_time=sabine_rt60 * 5, energy_threshold=power / num_rays * 1e-7)

    solution = solver.solve()

    rt60_threshold = solution.get_RT60(method="threshold")[0]
    rt60_schroeder = solution.get_RT60(method="schroeder")[0]
    sigma = 0.05 * sabine_rt60
    rt60_gaussian_schroeder = solution.get_RT60(kernel_func=gaussian(mu=5*sigma, sigma=sigma), method="schroeder")[0]

    df = pd.concat([df, pd.DataFrame([{
        "x_dim": x_dim, 
        "y_dim": y_dim, 
        "z_dim": z_dim, 
        "reflection_eff": reflection_eff, 
        "receiver_x": receiver_pos[0], 
        "receiver_y": receiver_pos[1], 
        "receiver_z": receiver_pos[2], 
        "volume": volume, 
        "surface_area": surface_area, 
        "sabine_rt60": sabine_rt60, 
        "eyring_rt60": eyring_rt60, 
        "rt60_threshold": rt60_threshold, 
        "rt60_schroeder": rt60_schroeder, 
        "rt60_gaussian_schroeder": rt60_gaussian_schroeder
    }])], ignore_index=True)

    for idx, receiver in enumerate(solution.receivers):
        hits = np.array(solution.receivers[idx].hits)
        if hits.size > 0 and hits.shape[1] >= 2:
            plt.scatter(hits[:, 0], np.log10(hits[:, 1]), label=f"Receiver {idx}", s=5, alpha=0.5)
            plt.xlabel('Time')
            plt.ylabel('Log10(Power)')
            plt.legend()
    plt.show()

df.to_csv("sabine_analysis.csv", index=False)
