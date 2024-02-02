"""LAMMPS has the options to use several internal units (which can be different
from the ones used in ase).  Mapping is therefore necessary.

See: https://lammps.sandia.gov/doc/units.html
 """

from . import units
from . import const as u

# !TODO add reduced Lennard-Jones units?

# NOTE: We assume a three-dimensional simulation here!
DIM = 3.0

UNITSETS = {}

UNITSETS["ASE"] = dict(
    mass=1.0 / units.kg,
    distance=1.0 / units.m,
    time=1.0 / units.second,
    energy=1.0 / units.J,
    velocity=units.second / units.m,
    force=units.m / units.J,
    pressure=1.0 / units.Pascal,
    charge=1.0 / units.C,
)

UNITSETS["real"] = dict(
    mass=u.gram_per_mole_si,
    distance=u.angstrom_si,
    time=u.femtosecond_si,
    energy=u.kcal_per_mole_si,
    velocity=u.angstrom_per_femtosecond_si,
    force=u.kcal_per_mole_angstrom_si,
    torque=u.kcal_per_mole_si,
    temperature=u.kelvin_si,
    pressure=u.atmosphere_si,
    dynamic_viscosity=u.poise_si,
    charge=u.e_si,
    dipole=u.electron_angstrom_si,
    electric_field=u.volt_per_angstrom_si,
    density=u.gram_si / u.centimeter_si ** DIM,
)

UNITSETS["metal"] = dict(
    mass=u.gram_per_mole_si,
    distance=u.angstrom_si,
    time=u.picosecond_si,
    energy=u.ev_si,
    velocity=u.angstrom_per_picosecond_si,
    force=u.ev_per_angstrom_si,
    torque=u.ev_si,
    temperature=u.kelvin_si,
    pressure=u.bar_si,
    dynamic_viscosity=u.poise_si,
    charge=u.e_si,
    dipole=u.electron_angstrom_si,
    electric_field=u.volt_per_angstrom_si,
    density=u.gram_si / u.centimeter_si ** DIM,
)

UNITSETS["si"] = dict(
    mass=u.kilogram_si,
    distance=u.meter_si,
    time=u.second_si,
    energy=u.joule_si,
    velocity=u.meter_per_second_si,
    force=u.newton_si,
    torque=u.joule_si,
    temperature=u.kelvin_si,
    pressure=u.pascal_si,
    dynamic_viscosity=u.pascal_si * u.second_si,
    charge=u.coulomb_si,
    dipole=u.coulomb_meter_si,
    electric_field=u.volt_per_meter_si,
    density=u.kilogram_si / u.meter_si ** DIM,
)

UNITSETS["cgs"] = dict(
    mass=u.gram_si,
    distance=u.centimeter_si,
    time=u.second_si,
    energy=u.erg_si,
    velocity=u.centimeter_per_second_si,
    force=u.dyne_si,
    torque=u.dyne_centimeter_si,
    temperature=u.kelvin_si,
    pressure=u.dyne_per_centimetersq_si,  # or barye =u. 1.0e-6 bars
    dynamic_viscosity=u.poise_si,
    charge=u.statcoulomb_si,  # or esu (4.8032044e-10 is a proton)
    dipole=u.statcoulomb_centimeter_si,  # =u. 10^18 debye,
    electric_field=u.statvolt_per_centimeter_si,  # or dyne / esu
    density=u.gram_si / (u.centimeter_si ** DIM),
)

UNITSETS["electron"] = dict(
    mass=u.amu_si,
    distance=u.bohr_si,
    time=u.femtosecond_si,
    energy=u.hartree_si,
    velocity=u.bohr_per_atu_si,
    force=u.hartree_per_bohr_si,
    temperature=u.kelvin_si,
    pressure=u.pascal_si,
    charge=u.e_si,  # multiple of electron charge (1.0 is a proton)
    dipole=u.debye_si,
    electric_field=u.volt_per_centimeter_si,
)

UNITSETS["micro"] = dict(
    mass=u.picogram_si,
    distance=u.micrometer_si,
    time=u.microsecond_si,
    energy=u.picogram_micrometersq_per_microsecondsq_si,
    velocity=u.micrometer_per_microsecond_si,
    force=u.picogram_micrometer_per_microsecondsq_si,
    torque=u.picogram_micrometersq_per_microsecondsq_si,
    temperature=u.kelvin_si,
    pressure=u.picogram_per_micrometer_microsecondsq_si,
    dynamic_viscosity=u.picogram_per_micrometer_microsecond_si,
    charge=u.picocoulomb_si,  # (1.6021765e-7 is a proton),
    dipole=u.picocoulomb_micrometer_si,
    electric_field=u.volt_per_micrometer_si,
    density=u.picogram_si / (u.micrometer_si) ** DIM,
)

UNITSETS["nano"] = dict(
    mass=u.attogram_si,
    distance=u.nanometer_si,
    time=u.nanosecond_si,
    energy=u.attogram_nanometersq_per_nanosecondsq_si,
    velocity=u.nanometer_per_nanosecond_si,
    force=u.attogram_nanometer_per_nanosecondsq_si,
    torque=u.attogram_nanometersq_per_nanosecondsq_si,
    temperature=u.kelvin_si,
    pressure=u.attogram_per_nanometer_nanosecondsq_si,
    dynamic_viscosity=u.attogram_per_nanometer_nanosecond_si,
    charge=u.e_si,  # multiple of electron charge (1.0 is a proton)
    dipole=u.electron_nanometer_si,
    electric_field=u.volt_si / u.nanometer_si,
    density=u.attogram_si / u.nanometer_si ** DIM,
)


def convert(value, quantity, fromunits, tounits):
    """Convert units between LAMMPS and ASE.

    :param value: converted value
    :param quantity: mass, distance, time, energy, velocity, force, torque,
    temperature, pressure, dynamic_viscosity, charge, dipole,
    electric_field or density
    :param fromunits: ASE, metal, real or other (see lammps docs).
    :param tounits: ASE, metal, real or other
    :returns: converted value
    :rtype:
    """
    return UNITSETS[fromunits][quantity] / UNITSETS[tounits][quantity] * value
