# this file is used to store the constants used in the code
deltaE = {
        1: -45.140551665, 3: -210.0485218888889, 4: -321.1987119, 5: -146.63024691666666, 6: -399.0110205833333, 
        7: -502.070125, 8: -879.0771215, 9: -1091.0652775, 11: -1275.295054, 12: -2131.9724644444445, 13: -2412.581311, 
        14: -787.3439924999999, 15: -1215.4995769047619, 16: -1705.5754946875, 17: -557.9141695, 19: -1544.3553605, 
        20: -1105.0024515, 21: -1420.574128, 22: -1970.9374273333333, 23: -2274.598644, 24: -2331.976294, 
        25: -2762.3960913793107, 26: -3298.6401545, 27: -3637.624857, 28: -4140.3502, 29: -5133.970898611111, 
        30: -5498.13054, 31: -2073.70436625, 32: -2013.83114375, 33: -463.783827, 34: -658.83885375, 35: -495.05260075, 
        37: -782.22601375, 38: -1136.1897344444444, 39: -1567.6510633333335, 40: -2136.8407, 41: -2568.946113, 
        42: -2845.9228975, 43: -3149.6645705, 44: -3640.458547, 45: -4080.81555, 46: -4952.347355, 
        47: -5073.703895555555, 48: -4879.3604305, 49: -2082.8865266666667, 50: -2051.94076125, 51: -2380.010715, 
        52: -2983.2449, 53: -3478.003375, 55: -1096.984396724138, 56: -969.538106, 72: -2433.925215, 73: -2419.015324, 
        74: -2872.458516, 75: -4684.01374, 76: -5170.37679, 77: -4678.720765, 78: -5133.04942, 79: -5055.7201, 
        80: -5791.21431, 81: -1412.194369, 82: -2018.85905225, 83: -2440.8732966666666
        }

elements = ["0", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
                "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", 
                "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
                "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", 
                "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", 
                "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
                "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", 
                "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", 
                "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
                "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", 
                "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg"]

ELEMENTTABLE={'H': 1,
    'He': 2,  'Li': 3,  'Be': 4,  'B': 5,   'C': 6,   'N': 7,   'O': 8,   'F': 9,   'Ne': 10,  'Na': 11,
    'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,  'S': 16,  'Cl': 17, 'Ar': 18, 'K': 19,  'Ca': 20,  'Sc': 21,
    'Ti': 22, 'V': 23,  'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,  'Ga': 31,
    'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39,  'Zr': 40,  'Nb': 41, 
    'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,  'Sb': 51, 
    'Te': 52, 'I': 53,  'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,  'Pm': 61,
    'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,  'Lu': 71, 
    'Hf': 72, 'Ta': 73, 'W': 74,  'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,  'Tl': 81, 
    'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,  'Pa': 91, 
    'U': 92,  'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
    'No': 102,'Lr': 103,'Rf': 104,'Db': 105,'Sg': 106,'Bh': 107,'Hs': 108,'Mt': 109,'Ds': 110,'Rg': 111,
    'Uub': 112
    }

ELEMENTTABLE_2 = {1: 'H', 
    2: 'He',     3: 'Li',   4: 'Be',   5: 'B',    6: 'C',    7: 'N',   8: 'O',     9: 'F',   10: 'Ne',  11: 'Na', 
    12: 'Mg',   13: 'Al',  14: 'Si',  15: 'P',   16: 'S',   17: 'Cl',  18: 'Ar',  19: 'K',   20: 'Ca',  21: 'Sc', 
    22: 'Ti',   23: 'V',   24: 'Cr',  25: 'Mn',  26: 'Fe',  27: 'Co',  28: 'Ni',  29: 'Cu',  30: 'Zn',  31: 'Ga', 
    32: 'Ge',   33: 'As',  34: 'Se',  35: 'Br',  36: 'Kr',  37: 'Rb',  38: 'Sr',  39: 'Y',   40: 'Zr',  41: 'Nb', 
    42: 'Mo',   43: 'Tc',  44: 'Ru',  45: 'Rh',  46: 'Pd',  47: 'Ag',  48: 'Cd',  49: 'In',  50: 'Sn',  51: 'Sb', 
    52: 'Te',   53: 'I',   54: 'Xe',  55: 'Cs',  56: 'Ba',  57: 'La',  58: 'Ce',  59: 'Pr',  60: 'Nd',  61: 'Pm', 
    62: 'Sm',   63: 'Eu',  64: 'Gd',  65: 'Tb',  66: 'Dy',  67: 'Ho',  68: 'Er',  69: 'Tm',  70: 'Yb',  71: 'Lu', 
    72: 'Hf',   73: 'Ta',  74:  'W',  75: 'Re',  76: 'Os',  77: 'Ir',  78: 'Pt',  79: 'Au',  80: 'Hg',  81: 'Tl', 
    82: 'Pb',   83: 'Bi',  84: 'Po',  85: 'At',  86: 'Rn',  87: 'Fr',  88: 'Ra',  89: 'Ac',  90: 'Th',  91: 'Pa', 
    92: 'U',    93: 'Np',  94: 'Pu',  95: 'Am',  96: 'Cm',  97: 'Bk',  98: 'Cf',  99: 'Es', 100: 'Fm', 101: 'Md', 
    102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 
    112: 'Uub'
    }

ELEMENTMASSTABLE={  1:1.008,2:4.002,3:6.941,4:9.012,5:10.811,6:12.011,
                            7:14.007,8:15.999,9:18.998,10:20.18,11:22.99,12:24.305,
                            13:26.982,14:28.086,15:30.974,16:32.065,17:35.453,
                            18:39.948,19:39.098,20:40.078,21:44.956,22:47.867,
                            23:50.942,24:51.996,25:54.938,26:55.845,27:58.933,
                            28:58.693,29:63.546,30:65.38,31:69.723,32:72.64,33:74.922,
                            34:78.96,35:79.904,36:83.798,37:85.468,38:87.62,39:88.906,
                            40:91.224,41:92.906,42:95.96,43:98,44:101.07,45:102.906,46:106.42,
                            47:107.868,48:112.411,49:114.818,50:118.71,51:121.76,52:127.6,
                            53:126.904,54:131.293,55:132.905,56:137.327,57:138.905,58:140.116,
                            59:140.908,60:144.242,61:145,62:150.36,63:151.964,64:157.25,65:158.925,
                            66:162.5,67:164.93,68:167.259,69:168.934,70:173.054,71:174.967,72:178.49,
                            73:180.948,74:183.84,75:186.207,76:190.23,77:192.217,78:195.084,
                            79:196.967,80:200.59,81:204.383,82:207.2,83:208.98,84:210,85:210,86:222,
                            87:223,88:226,89:227,90:232.038,91:231.036,92:238.029,93:237,94:244,
                            95:243,96:247,97:247,98:251,99:252,100:257,101:258,102:259,103:262,104:261,105:262,106:266}

ELEMENTMASSTABLE_2 = {'H': 1.008, 'He': 4.002, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 'C': 12.011,
                            'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.305,
                            'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.065, 'Cl': 35.453,
                            'Ar': 39.948, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867,
                            'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933,
                            'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.64, 'As': 74.922,
                            'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906,
                            'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.96, 'Tc': 98, 'Ru': 101.07, 'Rh': 102.906, 'Pd': 106.42,
                            'Ag': 107.868, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6,
                            'I': 126.904, 'Xe': 131.293, 'Cs': 132.905, 'Ba': 137.327, 'La': 138.905, 'Ce': 140.116,
                            'Pr': 140.908, 'Nd': 144.242, 'Pm': 145, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.925,
                            'Dy': 162.5, 'Ho': 164.93, 'Er': 167.259, 'Tm': 168.934, 'Yb': 173.054, 'Lu': 174.967, 'Hf': 178.49,
                            'Ta': 180.948, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084,
                            'Au': 196.967, 'Hg': 200.59, 'Tl': 204.383, 'Pb': 207.2, 'Bi': 208.98, 'Po': 210, 'At': 210, 'Rn': 222,
                            'Fr': 223, 'Ra': 226, 'Ac': 227, 'Th': 232.038, 'Pa': 231.036, 'U': 238.029, 'Np': 237, 'Pu': 244,
                            'Am': 243, 'Cm': 247, 'Bk': 247, 'Cf': 251, 'Es': 252, 'Fm': 257, 'Md': 258, 'No': 259, 'Lr': 262}

# The following definitions are all given in SI and are excerpted from the
# kim_units.cpp file created by Prof. Ellad B. Tadmor (UMinn) distributed with
# LAMMPS. Note that these values do not correspond to any official CODATA set
# already released, but rather to the values used internally by LAMMPS.
#
# Source: https://physics.nist.gov/cuu/Constants/Table/allascii.txt

from numpy import sqrt

# Constants
boltz_si = 1.38064852e-23  # [J K^-1] Boltzmann's factor (NIST value)
Nav = mole_si = 6.022140857e23  # [unitless] Avogadro's number
me_si = 9.10938356e-31  # [kg] electron rest mass
e_si = 1.6021766208e-19  # [C] elementary charge

# Distance units
meter_si = 1.0
bohr_si = (
    5.2917721067e-11
)  # [m] Bohr unit (distance between nucleus and electron in H)
angstrom_si = 1e-10  # [m] Angstrom
centimeter_si = 1e-2  # [m] centimeter
micrometer_si = 1e-6  # [m] micrometer (micron)
nanometer_si = 1e-9  # [m] nanometer

# Mass units
kilogram_si = 1.0
amu_si = gram_per_mole_si = 1e-3 / Nav  # [kg] gram per mole i.e. amu
gram_si = 1e-3  # [kg] gram
picogram_si = 1e-15  # [kg] picogram
attogram_si = 1e-21  # [kg[ attogram

# Time units
second_si = 1.0
atu_si = 2.418884326509e-17  # [s] atomic time unit
# ( = hbar/E_h where E_h is the  Hartree energy) (NIST value)
atu_electron_si = atu_si * sqrt(
    amu_si / me_si
)  # [s] atomic time unit used in electron system
microsecond_si = 1e-6  # [s] microsecond
nanosecond_si = 1e-9  # [s] nanosecond
picosecond_si = 1e-12  # [s] picosecond
femtosecond_si = 1e-15  # [s] femtosecond

# Density units
gram_per_centimetercu_si = (
    gram_si / centimeter_si ** 3
)  # [kg/m^3] gram/centimeter^3
amu_per_bohrcu_si = amu_si / bohr_si ** 3  # [kg/m^3] amu/bohr^3
picogram_per_micrometercu_si = (
    picogram_si / micrometer_si ** 3
)  # [kg/m^3] picogram/micrometer^3
attogram_per_nanometercu_si = (
    attogram_si / nanometer_si ** 3
)  # [kg/m^3] attogram/nanometer^3

# Energy/torque units
joule_si = 1.0
kcal_si = (
    4184.0
)  # [J] kilocalorie (heat energy involved in warming up one kilogram of
# water by one degree Kelvin)
ev_si = (
    1.6021766208e-19
)  # [J] electron volt (amount of energy gained or lost by the
# charge of a single electron moving across an electric
# potential difference of one volt.) (NIST value)
hartree_si = (
    4.359744650e-18
)  # [J] Hartree (approximately the electric potential energy
# of the hydrogen atom in its ground state) (NIST value)
kcal_per_mole_si = kcal_si / Nav  # [J] kcal/mole
erg_si = 1e-7  # [J] erg
dyne_centimeter_si = 1e-7  # [J[ dyne*centimeter
picogram_micrometersq_per_microsecondsq_si = (
    picogram_si * micrometer_si ** 2 / microsecond_si ** 2
)  # [J] picogram*micrometer^2/microsecond^2
attogram_nanometersq_per_nanosecondsq_si = (
    attogram_si * nanometer_si ** 2 / nanosecond_si ** 2
)  # [J] attogram*nanometer^2/nanosecond^2

# Velocity units
meter_per_second_si = 1.0
angstrom_per_femtosecond_si = (
    angstrom_si / femtosecond_si
)  # [m/s] Angstrom/femtosecond
angstrom_per_picosecond_si = (
    angstrom_si / picosecond_si
)  # [m/s] Angstrom/picosecond
micrometer_per_microsecond_si = (
    micrometer_si / microsecond_si
)  # [m/s] micrometer/microsecond
nanometer_per_nanosecond_si = (
    nanometer_si / nanosecond_si
)  # [m/s] nanometer/nanosecond
centimeter_per_second_si = centimeter_si  # [m/s] centimeter/second
bohr_per_atu_si = bohr_si / atu_electron_si  # [m/s] bohr/atu

# Force units
newton_si = 1.0
kcal_per_mole_angstrom_si = (
    kcal_per_mole_si / angstrom_si
)  # [N] kcal/(mole*Angstrom)
ev_per_angstrom_si = ev_si / angstrom_si  # [N] eV/Angstrom
dyne_si = dyne_centimeter_si / centimeter_si  # [N] dyne
hartree_per_bohr_si = hartree_si / bohr_si  # [N] hartree/bohr
picogram_micrometer_per_microsecondsq_si = (
    picogram_si * micrometer_si / microsecond_si ** 2
)  # [N] picogram*micrometer/microsecond^2
attogram_nanometer_per_nanosecondsq_si = (
    attogram_si * nanometer_si / nanosecond_si ** 2
)  # [N] attogram*nanometer/nanosecond^2

# Temperature units
kelvin_si = 1.0

# Pressure units
pascal_si = 1.0
atmosphere_si = 101325.0  # [Pa] standard atmosphere (NIST value)
bar_si = 1e5  # [Pa] bar
dyne_per_centimetersq_si = (
    dyne_centimeter_si / centimeter_si ** 3
)  # [Pa] dyne/centimeter^2
picogram_per_micrometer_microsecondsq_si = picogram_si / (
    micrometer_si * microsecond_si ** 2
)  # [Pa] picogram/(micrometer*microsecond^2)
attogram_per_nanometer_nanosecondsq_si = attogram_si / (
    nanometer_si * nanosecond_si ** 2
)  # [Pa] attogram/(nanometer*nanosecond^2)

# Viscosity units
poise_si = 0.1  # [Pa*s] Poise
amu_per_bohr_femtosecond_si = amu_si / (
    bohr_si * femtosecond_si
)  # [Pa*s] amu/(bohr*femtosecond)
picogram_per_micrometer_microsecond_si = picogram_si / (
    micrometer_si * microsecond_si
)  # [Pa*s] picogram/(micrometer*microsecond)
attogram_per_nanometer_nanosecond_si = attogram_si / (
    nanometer_si * nanosecond_si
)  # [Pa*s] attogram/(nanometer*nanosecond)

# Charge units
coulomb_si = 1.0
echarge_si = e_si  # [C] electron charge unit
statcoulomb_si = (
    e_si / 4.8032044e-10
)  # [C] Statcoulomb or esu (value from LAMMPS units documentation)
picocoulomb_si = 1e-12  # [C] picocoulomb

# Dipole units
coulomb_meter_si = 1
electron_angstrom_si = echarge_si * angstrom_si  # [C*m] electron*angstrom
statcoulomb_centimeter_si = (
    statcoulomb_si * centimeter_si
)  # [C*m] statcoulomb*centimeter
debye_si = 1e-18 * statcoulomb_centimeter_si  # [C*m] Debye
picocoulomb_micrometer_si = (
    picocoulomb_si * micrometer_si
)  # [C*m] picocoulomb*micrometer
electron_nanometer_si = echarge_si * nanometer_si  # [C*m] electron*nanometer

# Electric field units
volt_si = 1.0
volt_per_meter_si = 1
volt_per_angstrom_si = 1.0 / angstrom_si  # [V/m] volt/angstrom
statvolt_per_centimeter_si = erg_si / (
    statcoulomb_si * centimeter_si
)  # [V/m] statvolt/centimeter
volt_per_centimeter_si = 1.0 / centimeter_si  # [V/m] volt/centimeter
volt_per_micrometer_si = 1.0 / micrometer_si  # [V/m] volt/micrometer
volt_per_nanometer_si = 1.0 / nanometer_si  # [V/m] volt/nanometer
