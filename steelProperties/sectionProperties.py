import numpy as np

class Material:
    """
    Class to represent the material properties of the steel.

    Attributes:
    ----------
    - Fy (float): Yield strength of the material (MPa).
    - E (float): Modulus of elasticity (MPa).
    - gamma (float): Unit weight of the material (kN/m³).
    - Ry (float): Overstrength factor (dimensionless).
    """
    def __init__(self, Fy, E, gamma, Ry):
        self.Fy = Fy      # Yield strength
        self.E = E        # Modulus of elasticity
        self.gamma = gamma  # Unit weight
        self.Ry = Ry      # Overstrength factor


class WSection:
    """
    Class to represent a W-section (wide flange) steel beam.

    Attributes:
    ----------
    - bf (float): Flange width (mm).
    - tf (float): Flange thickness (mm).
    - h (float): Height of the web section (mm).
    - tw (float): Web thickness (mm).
    - material (Material): Material properties of the section.

    Methods:
    -------
    - area(): Calculates the gross area of the section.
    - y_center_mass(): Calculates the centroid (center of mass) of half the beam.
    - plastic_section_modulus(): Calculates the plastic section modulus (Zx).
    - summary(): Prints a summary of the section properties.
    - calculate_expected_moment_capacity(): Calculates the expected moment capacity (Mpr).
    - calculate_nominal_moment_capacity(Pu): Calculates the nominal moment capacity (Mn) considering axial load (Pu).
    """
    def __init__(self, bf, tf, h, tw, material, units=None):
        self.bf = bf       # Flange width
        self.tf = tf       # Flange thickness
        self.h = h         # Total height
        self.tw = tw       # Web thickness
        self.material = material
        
        if units is None:
            self.units={'length':1,
                        'force':1}
        elif not isinstance(units, dict):
            raise ValueError('Units must be a dictionary')
        else:
            self.units=units

        # Calculated properties
        self.Ag = self.area()
        self.Ix, self.Iy=self.calculate_moment_inercia()
        self.y_cm, self.x_cm = self.y_center_mass()
        self.Sx, self.Sy = self.section_modulus()
        self.Zx, self.Zy = self.plastic_section_modulus()

    def __str__(self):
        """
        Returns a formatted string representation of the W-section.
        """
        return f"W{self.bf}x{self.tf}-{self.h}x{self.tw}"

    def area(self):
        """
        Calculates the gross area of the W-section.

        Returns:
        -------
        - Ag (float): Gross area (mm²).
        """
        flange_area = 2 * self.bf * self.tf
        web_area = self.h * self.tw
        return flange_area + web_area

    def calculate_moment_inercia(self):
        Ix=(self.bf*(self.h+self.tf*2)**3/12)-((self.bf-self.tw)*self.h**3/12)
        Iy=(2)*(self.tf*self.bf**3/12)+(self.h*self.tw**3/12)
        return Ix, Iy

    def y_center_mass(self):
        """
        Calculates the centroid (center of mass) of half the W-section.

        Returns:
        -------
        - y_cm (float): Centroid of half the beam (mm).
        """
        # Using your original formula for half the beam's centroid calculation.
        area_momentum_y = ((self.h / 2 * self.tw) * (self.h / 4) + (self.bf * self.tf) * (self.h / 2 + self.tf / 2))
        y_cm = area_momentum_y / (self.Ag / 2)
        
        area_momentum_x=2*((self.tf*self.bf/2)*(self.bf/4))+((self.h*self.tw/2)*(self.tw/4))
        x_cm=area_momentum_x/(self.Ag/2)
        
        return y_cm, x_cm
    
    def section_modulus(self):
        c_x=self.h/2+self.tf
        c_y=self.bf/2
        Sx=self.Ix/c_x
        Sy=self.Iy/c_y
        return Sx, Sy

    def plastic_section_modulus(self):
        """
        Calculates the plastic section modulus (Zx).

        Returns:
        -------
        - Zx (float): Plastic section modulus (mm³).
        """
        Zx = self.y_cm * self.Ag
        Zy=self.x_cm*self.Ag
        return Zx, Zy

    def calculate_expected_moment_capacity(self, verbose=False):
        """
        Calculates the expected moment capacity (Mpr) of the section.

        Returns:
        -------
        - Mpr (float): Expected moment capacity (Consistent Units).
        """
        Zx = self.Zx
        Fy = self.material.Fy
        Ry = self.material.Ry
        Mpr = Ry * Fy * Zx
        
        if verbose is True:
            print(f"Beam expected moment capacity is: {np.round(Mpr/(self.units['force']*self.units['length']),1)}")
        
        return Mpr

    def calculate_nominal_moment_capacity(self, Pu):
        """
        Calculates the nominal moment capacity (Mn) considering axial load (Pu).

        Parameters:
        ----------
        - Pu (float): Axial load (Consistent Units).

        Returns:
        -------
        - Mn (float): Nominal moment capacity (Consistent Units).
        """
        Zx = self.Zx
        Fy = self.material.Fy
        Ag = self.Ag
        Mn = Zx * (Fy - Pu / Ag)
        return Mn
    
    def calculate_shear_capacity(self):
        Aw=self.h*self.tw
        Vn=0.60*self.material.Fy*Aw
        phiVn=0.90*Vn
        Vp=0.60*self.material.Fy*Aw*self.material.Ry
        return phiVn, Vn, Vp

    def summary(self):
        """
        Prints a summary of the W-section properties.
        """
        print("WSection Properties:")
        print(f"Gross Area (Ag): {np.round(self.Ag/(self.units['lenght'])**2, 3)}")
        
        print(f'Moment of inercia in the strong axis is: {np.round(self.Ix/(self.units['length'])**4,0)}')
        print(f'Moment of inercia in the weak axis is: {np.round(self.Iy/(self.units['length'])**4,0)}')
        
        print(f'Section Modulus in the strong axis (Sx): {np.round(self.Sx/(self.units['length'])**3,0)}')
        print(f'Section Modulus in the weak axis (Sy) : {np.round(self.Sy/(self.units['length'])**3,0)}')
        
        print(f"Centroid of Half the Beam (y_cm): {np.round(self.y_cm/(self.units['length']), 3)}")
        print(f"Plastic Section Modulus in the strong axis (Zx): {np.round(self.Zx/(self.units['length'])**3, 3)}")
        
        print(f"Centroid of Half the Beam (x_cm): {np.round(self.x_cm/(self.units['length']), 3)}")
        print(f"Plastic Section Modulus in the weak axis (Zy): {np.round(self.Zy/(self.units['length'])**3, 3)}")

class panelZone:
    def __init__(self, beamObject, columnObject, units=None):
        self.beam=beamObject
        self.column=columnObject
        
        if units is None:
            self.units={'length':1,
                        'force':1}  
        elif not isinstance(units, dict):
            raise ValueError('Units must be a dictionary')
        else:
            self.units=units
        
    def _tensionCapacity(self):
        
        # Calculate minimum thickness required by code
        tcf_limit=[]
        tcf_limit.append(self.beam.bf/6)
        tcf_limit.append(0.40*(1.80*self.beam.bf*self.beam.tf*(self.beam.material.Fy*self.beam.material.Ry)/(self.column.material.Fy*self.column.material.Ry))**0.5)
        
        # Set the units
        tcf_limit=tcf_limit
        
        # Calculate the demand and capacity
        phiRn=0.90*(6.25*self.column.tf**2*self.column.material.Fy*self.column.material.Ry)
        Tu=1.80*self.beam.bf*self.beam.tf*self.beam.material.Fy*self.beam.material.Ry
        
        ratio=Tu/phiRn
        
        results={'tcf_limit':tcf_limit, 'phiRn':phiRn, 'Tu':Tu, 'ratio':ratio}
        
        print(f'The minimum thickness to avoid tension flange rupture is: {np.round(tcf_limit[0],2)}')
        print(f'The column flange capacity is: {np.round(phiRn,2)}')
        print(f'The beam flange demand is: {np.round(Tu,2)}')
        print(f'The ratio between demand and capacity is: {np.round(ratio,2)}')
        
        return results
        
    

if __name__ == "__main__":
    # Define material properties (example values)
    Gr50 = Material(Fy=355, E=210000, gamma=78.5, Ry=1.1)

    # Define W-section dimensions (example values in mm)
    W1 = WSection(bf=300, tf=22, h=350, tw=12, material=Gr50)

    # Display summary of the section properties
    W1.summary()

    # Calculate and print expected moment capacity
    Mpr = W1.calculate_expected_moment_capacity()
    print(f"Expected Moment Capacity (Mpr): {np.round(Mpr, 3)}")

    # Calculate and print nominal moment capacity with an example axial load (Pu)
    Pu = 100000  # Example axial load in N
    Mn = W1.calculate_nominal_moment_capacity(Pu)
    print(f"Nominal Moment Capacity (Mn) with Pu={Pu}: {np.round(Mn, 3)}")



