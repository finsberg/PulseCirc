import circulation
import numpy as np

#%%
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
        self.p0=0
        self.r0=params["radius"]
        self.t0=params["thickness"]
        self.E=params["young_modulus"]
        self.nu=params["poisson_ratio"]
        
        self.p=self.p0
        self.r=self.r0
        self.t=self.t0
        
    def _compute_pressure_from_activation(self, activation: float):
        self.p=activation*2

    def inflate(self, activation: float) -> float:
        self._compute_pressure_from_activation(activation)
        
        p=self.p
        r0=self.r0
        t0=self.t0
        E=self.E
        nu=self.nu
        
        sigma=p*r0/(2*t0)
        eps=sigma/E
        dr=eps*r0
        r=r0+dr
    
        self.r=r
        self.t=t0-nu*dr
    
    def get_pressure(self):
        return self.p
    
    def get_volume(self):
        r=self.r
        volume=4/3*np.pi*r**3 
        return volume

    def dV_dP(self): 
        pass
        
    def save(self, t: float):
        pass
        #reults_u, p = self.problem.state.split(deepcopy=True)
        #reults_u.t=t+1
        #with dolfin.XDMFFile(outname.as_posix()) as xdmf:
        #    xdmf.write_checkpoint(reults_u, "u", float(t+1), dolfin.XDMFFile.Encoding.HDF5, True)


#%%
activation=np.arange(0,10,1)
time=np.arange(0,1,.1)
0
# Units [mm], [N], [MPa]
params={"radius": 20,"thickness": 2,"young_modulus": 50, "poisson_ratio0": 0.3}
model=ThinSphere(params)
#%%