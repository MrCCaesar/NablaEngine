"""Perception sensors on the engine's scenes (docs/sensors.md): batched GPU ray casting
against the terrain (boxes, heightfields, an optional floor plane), per-world objects (boxes) and the robot's visual
meshes (posed per world from the body poses), with the hit geometry's id; the depth camera (the G1's RealSense D435i),
LiDAR (Livox MID-360) and height-scan patterns on top of it. Independent of the physics kernels (its own library)."""
