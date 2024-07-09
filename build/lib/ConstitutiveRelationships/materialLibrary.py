# Material library for the streamlit app

# import dependencies
import ConstitutiveRelationships as cr
from baseUnits import kgf, cm, ksi


A36=cr.uniaxialBilinealSteel('A36', 36*ksi, 1.50*36*ksi)
A572=cr.uniaxialBilinealSteel('A572', 50*ksi, 1.10*36*ksi)
A706Gr60=cr.uniaxialBilinealSteel('A706Gr60', 60*ksi, 1.25*36*ksi)
fc210uc=cr.uniaxialUnconfinedConcrete('fc210uc', 210*kgf/cm**2)
fc240uc=cr.uniaxialUnconfinedConcrete('fc240uc', 240*kgf/cm**2)
fc280uc=cr.uniaxialUnconfinedConcrete('fc280uc', 280*kgf/cm**2)
fc350uc=cr.uniaxialUnconfinedConcrete('fc350uc', 350*kgf/cm**2)
