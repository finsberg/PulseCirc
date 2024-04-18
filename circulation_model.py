#%%
from scipy.integrate import solve_ivp

class CirculationModel():
    def __init__(self, params: dict = None):
        
        default_params = self.get_default_params()
        self.parameters = {key: params.get(key, default_params[key]) for key in default_params} if params else default_params
        
        self.aortic_pressure=self.parameters["diastolic_pressure"]
        self.aortic_pressure_derivation=0  
        
    def compute_outflow(self, pressure_current: float, pressure_old: float, dt: float)-> float:
        """Compute Q, Q_r, and Q_c for given LV pressure and time step dt."""
        valve_pressure = self.parameters["aortic_valve_pressure"]
        
        if pressure_current>valve_pressure:
            p_dia = self.parameters["diastolic_pressure"]
            R_sys = self.parameters["systemic_resistance"]
            R_ao = self.parameters["aortic_resistance"]
            C_sys = self.parameters["systemic_compliance"]

            circ_solution = solve_ivp(self.windkessel_3elements, [0, dt], [self.aortic_pressure, self.aortic_pressure_derivation],t_eval=[0, dt], args=(pressure_old,pressure_current))

            self.aortic_pressure=circ_solution.y[0][1]
            self.aortic_pressure_derivation=circ_solution.y[1][1]
            Q=(pressure_current-self.aortic_pressure)/R_ao
        else:
            Q=0 
            
        return Q
    
    def compute_outflow_derivation(self, pressure_current: float, pressure_old: float, dt: float, epsilon: float = 0.001)-> float:
        """Calculate the derivative of flow Q with respect to the pressure P at a given pressure using a finite difference method."""
        
        aortic_pressure_backup=self.aortic_pressure
        aortic_pressure_derivation_backup=self.aortic_pressure_derivation 
        
        Q_i= self.compute_outflow(pressure_current, pressure_old, dt)                   
        Q_f = self.compute_outflow(pressure_current + epsilon, pressure_old , dt)   
        
        # Use the finite difference approximation for the derivative
        dQ_dP = (Q_f - Q_i) / epsilon
        
        self.aortic_pressure=aortic_pressure_backup
        self.aortic_pressure_derivation=aortic_pressure_derivation_backup
        
        return dQ_dP
    
    def windkessel_3elements(self,t,y,p_old,p_current):
        
        # Defining WK3 function based on scipy.integrate.solve_ivp
        # The main equations are, with p_{ao} and its derivatives are unkowns:
        # 1. Q = \frac{p_{lv} - p_{ao}}{R_{ao}}
        # 2. Q_R = \frac{p_{ao}}{R_{circ}}
        # 3. Q_C = C_{circ} \cdot \frac{dp_{ao}}{dt}
        # 4. Q = Q_R + Q_C
        # 5. \frac{dp_{ao}}{dt} = y[1]
        # 6. \frac{d^2p_{ao}}{dt^2} = \frac{Q - Q_R - Q_C}{C_{circ}}
        
        p_dia = self.parameters["diastolic_pressure"]
        R_sys = self.parameters["systemic_resistance"]
        R_ao = self.parameters["aortic_resistance"]
        C_sys = self.parameters["systemic_compliance"]
        
        p_ao = y[0]
        dp_ao_dt = y[1]

        # Calculating flows
        p_lv_interpolated=p_old + (p_current - p_old) * t
        Q = (p_lv_interpolated - p_ao) / R_ao
        Q_R = (p_ao-p_dia) / R_sys
        Q_C = C_sys * dp_ao_dt

        # Conservation of flow
        dQ_C_dt = (Q - Q_R - Q_C) / C_sys
        d2p_ao_dt2=dQ_C_dt

        return [dp_ao_dt, d2p_ao_dt2]

    @staticmethod
    def get_default_params():
        """Return the default parameters for the circulation model."""
        return {
            "aortic_resistance": 1,
            "systemic_resistance": 10,
            "systemic_compliance": 5,
            "aortic_pressure": 10,
            "diastolic_pressure": 10,
            "aortic_valve_pressure": 10
        }

    def get_parameter(self, key: str) -> float:
        """Retrieve a parameter value by key."""
        return self.parameters.get(key, None)

    def set_parameter(self, key: str, value: float):
        """Set a parameter value by key."""
        if key in self.parameters:
            self.parameters[key] = value
        else:
            raise KeyError(f"Parameter {key} not found in the model.")
        
