"""
An example of how to use the ray tracing solver.
"""
from room import *
from ray_tracing import RayTracingSolver
import matplotlib.pyplot as plt

dim = 5

room = Room(reflection_eff=0.8, dimension=3, bounds={'x': (-dim, dim), 'y': (-dim, dim), 'z': (-dim, dim)})
room.add_surface(Ellipsoid(center=[0.5, -5, 0.2], radii=[2, 2, 2], reflection_eff=0.8))
room.add_surface(Ellipsoid(center=[2, 2, 2], radii=[2, 3, 3], reflection_eff=0.8))
sources = [Source(position=[0, 0, 0], power=10, num_rays=10000, dimension=3)]
receivers = [Receiver(position=[0, -2, -2], radii=[0.1, 0.1, 0.1], type="virtual"), 
             Receiver(position=[0, 2, 2], radii=[0.1, 0.1, 0.1], type="virtual"), 
             Receiver(position=[0, 2, -2], radii=[0.1, 0.1, 0.1], type="virtual"), 
             Receiver(position=[0, -2, 2], radii=[0.1, 0.1, 0.1], type="virtual")]
solver = RayTracingSolver(room, sources, receivers, max_time=10.0, energy_threshold=1e-10)

solution = solver.solve()

solver.plot(show_solution=True, num_rays_show=5, opacity=True)

def gaussian(mu, sigma):
    return lambda t: np.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

responses = solution.get_responses(kernel_func=gaussian(mu=0.05, sigma=0.01), aggregate=True)

for response in responses:
    plt.figure()
    plt.plot(response[0], response[1])
    plt.xlabel('Time (s)')
    plt.ylabel('Power (W)')
    plt.xlim(0, 1)
    plt.title('Room Impulse Response')
    plt.show()

print("RT60s")
print("Linear regression on log-hits: ", solution.get_RT60(kernel_func=None, method="threshold", aggregate=True))
print("Schroeder on hits: ", solution.get_RT60(kernel_func=None, method="schroeder", aggregate=True))
print("Schroeder with Gaussian kernel: ", solution.get_RT60(kernel_func=gaussian(mu=0.05, sigma=0.01), method="schroeder", aggregate=True))