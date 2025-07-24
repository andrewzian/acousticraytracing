# Acoustic Ray Tracing

This project utilizes ray tracing to model room acoustics, in particular to probe the room impulse response and reverberation time of different room geometries.

<figure style="text-align: center;">
  <img width="408" height="388" alt="Screenshot 2025-07-23 at 11 48 27 PM" src="https://github.com/user-attachments/assets/0e315ba9-acea-4352-8e06-cc6c323e4deb" style="display: block; margin: 0 auto;" width="400"/>
  <figcaption> </figcaption>
</figure>

<p align="center">
  <img src="https://github.com/user-attachments/assets/0e315ba9-acea-4352-8e06-cc6c323e4deb" alt="Screenshot 2025-07-23 at 11 48 27 PM" width="400"/>
</p>

<p align="center">
  <em>An example ray tracing visualization.</em>
</p>
    
_An example ray tracing visualization._


## Background

An obvious approach to take in modeling room acoustics is to define appropriate boundary conditions and solve the 3D wave equation, perhaps using finite-difference time-domain (FDTD) methods. However, such methods, which discretize space and time and approximate derivatives using finite differences, are extremely computationally intensive, with cost scaling with each of the three spatial dimensions as well as time. Fortunately, ray tracing offers an alternative approach that is accurate in the high-frequency regime, where wave phenomena are negligible.

[ISO 3382-2]([url](https://cdn.standards.iteh.ai/samples/36201/4a4d0dc848ac4d40bfe46e36c531afb5/ISO-3382-2-2008.pdf)) defines **impulse response** as “temporal evolution of the sound pressure observed at a point in a room as a result of the emission of a Dirac impulse at another point in the room,” and **reverberation time (RT60)** as “duration required for the space-averaged sound energy density in an enclosure to decrease by 60 dB after the source emission has stopped.” RT60 is a widely used metric in architectural acoustics to characterize the clarity or muddiness of sound in a space. For example, the RT60 in Stanford University's Bing Concert Hall is about 2.5 seconds. Historically, acoustical physicists developed empirical heuristics for estimating RT60. Among these were the [Sabine equation]([url](https://en.wikipedia.org/wiki/Reverberation#Sabine_equation)) and the [Eyring equation]([url](https://en.wikipedia.org/wiki/Reverberation#Eyring_equation)).

## Features

- `room.py`. Defines `Surface`, including children `PlaneWall`, `Ellipsoid`, and `Paraboloid`, which can be used to define room geometries. Also defines `Ray`, `Source`, and `Receiver`.
- `ray_tracing.py`. Provides `RayTracingSolver`, which implements ray tracing and does plotting, and `RayTracingSolution`, which is used to compute room impulse response and RT60.
- `sabine_analysis.py`. Implements Monte Carlo simulation of different rectangular room geometries and their simulated RT60s and Sabine equation RT60s. A dataset produced by this code can be found on [Kaggle](url).
- `ray_example.py`. Demonstrates how to define a room geometry and run the ray tracing simulation.

## Simulation Flow

Given as input a room geometry (positions and orientations of all surfaces), reflection efficiencies of all surfaces, the positions and sizes of all sounds sources and sound receivers in the room, and other parameters (e.g. number of rays to simulate, source sound power), the simulation outputs the path taken by each ray and a list of ray hits measured by each receiver. The list of receiver hits can then be convolved with a kernel to produce the room impulse response. The RT60 can be computed by one of three methods: linear regression on log-power of hits, [Schroeder integration]([url](https://pubs.aip.org/asa/jasa/article/157/2/R3/3333468/Schroeder-integration-for-sound-energy-decay)) on hits, and Schroeder integration with a kernel. A method for interactive plotting using Plotly is provided for visualization. 
