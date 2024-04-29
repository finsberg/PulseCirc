import numpy as np


# %%
class ThinSphere:
    """
    The stress (sigma) in a thin-walled spherical vessel due to internal pressure is given by:
    sigma = p * r / (2 * t)
    where p is the internal pressure, r is the radius, and t is the thickness of the sphere.
    Assuming linear elasticity epsilon = sigma / E =  Δr / r
    The change in thickness (Δt) can also be approximated using:
    Δt ≈ -nu * (pr / (2tE)) * t
    where nu is the Poisson's ratio of the material
    """

    def __init__(self, params):
        self.p = 0
        self.r = params["radius"]
        self.t = params["thickness"]
        self.E = params["young_modulus"]
        self.nu = params["poisson_ratio"]

    def inflate(self, pressure: float):
        r = self.r
        t = self.t
        E = self.E
        nu = self.nu
        # Calculate strain and change in radius
        sigma = pressure * r / (2 * t)
        eps = sigma / E
        dr = eps * r
        # Update the pressure, radius, and thickness
        self.r = r + dr
        self.t = t - nu * dr
        self.p = pressure

    def contract(self, activation: float):
        r = self.r
        t = self.t
        E = self.E
        nu = self.nu
        # Calculate strain and change in radius
        sigma = activation
        eps = sigma / E
        dr = eps * r
        # Update the radius, and thickness
        self.r = r - dr
        self.t = t - nu * dr

    def get_pressure(self):
        return self.p

    def get_volume(self):
        r = self.r
        volume = 4 / 3 * np.pi * r**3
        return volume

    def dv_dp(self):
        r0 = self.r
        t0 = self.t
        p0 = self.p
        v0 = self.get_volume()

        p = p0 * (1 + 1e-6)
        self.inflate(p)
        v = self.get_volume()

        dv_dp = (v - v0) / (p - p0)

        self.r = r0
        self.t = t0
        self.p = p0
        return dv_dp

    def save(self, t: float):
        pass


# %%
activation = np.linspace(0, 0.1, 100)
pressure = np.linspace(0, 0.015, 100)

# Units [mm], [N], [MPa]
params = {"radius": 50, "thickness": 5, "young_modulus": 50, "poisson_ratio": 0.3}
model = ThinSphere(params)
vols = []
pres = []
for a, p in zip(activation, pressure):
    model.contract(a)
    model.inflate(p)
    pres.append(model.get_pressure())
    vols.append(model.get_volume())

import matplotlib.pyplot as plt

plt.plot(vols, pres)
# %%
