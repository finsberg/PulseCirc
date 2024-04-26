#%%
import numpy as np

class ThinSphere:
    """
    Models the mechanical behavior of a thin-walled spherical vessel under internal pressure using linear elasticity.
    The stress (sigma) and deformation (change in radius and thickness) due to internal pressure are calculated.
    """

    def __init__(self, radius: float, thickness: float, young_modulus: float, poisson_ratio: float):
        """
        Initializes a ThinSphere object with the necessary material and geometric properties.

        Parameters:
        radius (float): Radius of the sphere.
        thickness (float): Thickness of the sphere wall.
        young_modulus (float): Young's modulus of the material.
        poisson_ratio (float): Poisson's ratio of the material.
        """
        self.pressure = 0
        self.activation = 0
        self.radius = radius
        self.thickness = thickness
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        
    def compute_volume(self, activation: float, pressure: float) -> float:
        """
        Computes the volume of the sphere after applying the given activation and internal pressure.

        Parameters:
        activation (float): External factor influencing material stress independently from internal pressure.
        pressure (float): Internal pressure applied to the sphere.

        Returns:
        float: New volume of the sphere after deformation.
        """
        current_radius = self.radius
        current_thickness = self.thickness

        # Calculate change in radius due to activation
        stress_activation = activation
        strain_activation = stress_activation / self.young_modulus
        change_in_radius_activation = strain_activation * current_radius

        # Update radius and thickness due to activation
        current_radius -= change_in_radius_activation
        current_thickness -= self.poisson_ratio * change_in_radius_activation

        # Calculate new stress from internal pressure and resulting strain
        stress_pressure = pressure * current_radius / (2 * current_thickness)
        strain_pressure = stress_pressure / self.young_modulus
        change_in_radius_pressure = strain_pressure * current_radius

        # Update radius and thickness due to internal pressure
        self.radius = current_radius + change_in_radius_pressure
        self.thickness -= self.poisson_ratio * change_in_radius_pressure
        self.pressure = pressure
        self.activation = activation

        # Calculate new volume
        new_volume = 4/3 * np.pi * self.radius**3
        return new_volume
        
    def get_pressure(self) -> float:

        return self.pressure
    
    def get_volume(self) -> float:

        return 4/3 * np.pi * self.radius**3

    def compute_volume_derivative(self, activation: float, pressure: float,  pressure_change = 0.01) -> float:
        """
        Computes the derivative of volume with respect to pressure by simulating a small pressure change.

        Parameters:
        activation (float): Activation level to be applied during the computation.
        pressure (float): Baseline pressure for which the derivative is to be calculated.

        Returns:
        float: Derivative of the volume with respect to pressure.
        """
        initial_radius = self.radius
        initial_thickness = self.thickness
        initial_pressure = self.pressure
        initial_activation = self.activation
        initial_volume = self.get_volume()
        
        # Apply a small pressure increase for derivative calculation
        new_volume = self.compute_volume(activation, pressure + pressure_change)
        volume_derivative = (new_volume - initial_volume) / 0.01
        
        # Restore original state
        self.radius = initial_radius
        self.thickness = initial_thickness
        self.pressure = initial_pressure
        self.activation = initial_activation
        
        return volume_derivative
        
    def save(self, filename: str):

        pass

# %% An Example :

# thin_sphere=ThinSphere(50,5,50,0.3)
# activation=np.linspace(0,0.1,200)
# pressure=np.linspace(0,.015,200)
# pres=[]
# vols=[]
# ts=[]
# for (a,p) in zip(activation,pressure):
#     thin_sphere.compute_volume(a,p)
#     pres.append(thin_sphere.get_pressure())
#     vols.append(thin_sphere.get_volume())
#     ts.append(thin_sphere.thickness)

# import matplotlib.pyplot as plt   
# plt.plot(vols,pres)
# # plt.plot(ts)