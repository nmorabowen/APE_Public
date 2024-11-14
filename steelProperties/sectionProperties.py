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
    - h (float): Total height of the section (mm).
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
    def __init__(self, bf, tf, h, tw, material):
        self.bf = bf       # Flange width
        self.tf = tf       # Flange thickness
        self.h = h         # Total height
        self.tw = tw       # Web thickness
        self.material = material

        # Calculated properties
        self.Ag = self.area()
        self.y_cm = self.y_center_mass()
        self.Zx = self.plastic_section_modulus()

    def __str__(self):
        """
        Returns a formatted string representation of the W-section.
        """
        return f"WSection: {self.bf}x{self.tf} - {self.h}x{self.tw}"

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

    def y_center_mass(self):
        """
        Calculates the centroid (center of mass) of half the W-section.

        Returns:
        -------
        - y_cm (float): Centroid of half the beam (mm).
        """
        # Using your original formula for half the beam's centroid calculation.
        area_momentum = ((self.h / 2 * self.tw) * (self.h / 4) + (self.bf * self.tf) * (self.h / 2 + self.tf / 2))
        y_cm = area_momentum / (self.Ag / 2)
        return y_cm

    def plastic_section_modulus(self):
        """
        Calculates the plastic section modulus (Zx).

        Returns:
        -------
        - Zx (float): Plastic section modulus (mm³).
        """
        Zx = self.y_cm * self.Ag
        return Zx

    def calculate_expected_moment_capacity(self):
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

    def summary(self):
        """
        Prints a summary of the W-section properties.
        """
        print("WSection Properties:")
        print(f"Gross Area (Ag): {np.round(self.Ag, 3)}")
        print(f"Centroid of Half the Beam (y_cm): {np.round(self.y_cm, 3)}")
        print(f"Plastic Section Modulus (Zx): {np.round(self.Zx, 3)}")


if __name__ == "__main__":
    # Define material properties (example values)
    Gr50 = Material(Fy=355, E=210000, gamma=78.5, Ry=1.1)

    # Define W-section dimensions (example values in mm)
    W1 = WSection(bf=180, tf=15, h=450, tw=8, material=Gr50)

    # Display summary of the section properties
    W1.summary()

    # Calculate and print expected moment capacity
    Mpr = W1.calculate_expected_moment_capacity()
    print(f"Expected Moment Capacity (Mpr): {np.round(Mpr, 3)}")

    # Calculate and print nominal moment capacity with an example axial load (Pu)
    Pu = 100000  # Example axial load in N
    Mn = W1.calculate_nominal_moment_capacity(Pu)
    print(f"Nominal Moment Capacity (Mn) with Pu={Pu}: {np.round(Mn, 3)}")

