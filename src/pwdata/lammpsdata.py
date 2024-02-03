import re
import numpy as np
from tqdm import tqdm
from collections import Counter
from image import Image
from calculators.unitconvert_lmps import convert
from calculators.const import ELEMENTMASSTABLE

class LMP(object):
    def __init__(self, lmp_file, atom_names: list[str] = None, units: str = 'metal', style: str = 'atomic', sort_by_id: bool = True) -> None:
        """Method which reads a LAMMPS data file.

        Parameters
        ----------
        lmp_file : file | str
            File from which data should be read.
        atom_names : dict[int, int], optional
            Mapping from LAMMPS atom types (typically starting from 1) to atomic
            numbers. If None, if there is the "Masses" section, atomic numbers are
            guessed from the atomic masses. Otherwise, atomic numbers of 1 (H), 2
            (He), etc. are assigned to atom types of 1, 2, etc. Default is None.
        sort_by_id : bool, optional
            Order the particles according to their id. Might be faster to set it
            False. Default is True.
        units : str, optional
            `LAMMPS units <https://docs.lammps.org/units.html>`__. Default is
            'metal'.
        style : {'atomic', 'charge', 'full'} etc., optional
            `LAMMPS atom style <https://docs.lammps.org/atom_style.html>`__.
            If None, `style` is guessed in the following priority (1) comment
            after `Atoms` (2) length of fields (valid only `atomic` and `full`).
            Default is None.
        """
        self.image_list:list[Image] = []
        # self.number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
        self.atom_names = atom_names
        self.load_lmp_file(lmp_file, units, style, sort_by_id)        # load lammps init file, Adapted from ase: http://wiki.fysik.dtu.dk/ase

        assert len(self.image_list) > 0, "No system loaded!"

    def get(self):
        return self.image_list
    
    def load_lmp_file(self, lmp_file, units, style, sort_by_id):
        # begin read_lammps_data
        comment = None
        atom_nums = None
        # N_types = None
        atom_names = []     # if self.atom_names is None, atom_names will be guessed from the atomic masses
        xlo = None
        xhi = None
        ylo = None
        yhi = None
        zlo = None
        zhi = None
        xy = None
        xz = None
        yz = None
        pos_in = {}
        travel_in = {}
        mol_id_in = {}
        charge_in = {}
        mass_in = {}
        vel_in = {}
        bonds_in = []
        angles_in = []
        dihedrals_in = []
        sections = [
            "Atoms",
            "Velocities",
            "Masses",
            "Charges",
            "Ellipsoids",
            "Lines",
            "Triangles",
            "Bodies",
            "Bonds",
            "Angles",
            "Dihedrals",
            "Impropers",
            "Impropers Pair Coeffs",
            "PairIJ Coeffs",
            "Pair Coeffs",
            "Bond Coeffs",
            "Angle Coeffs",
            "Dihedral Coeffs",
            "Improper Coeffs",
            "BondBond Coeffs",
            "BondAngle Coeffs",
            "MiddleBondTorsion Coeffs",
            "EndBondTorsion Coeffs",
            "AngleTorsion Coeffs",
            "AngleAngleTorsion Coeffs",
            "BondBond13 Coeffs",
            "AngleAngle Coeffs",
        ]
        header_fields = [
            "atoms",
            "bonds",
            "angles",
            "dihedrals",
            "impropers",
            "atom types",
            "bond types",
            "angle types",
            "dihedral types",
            "improper types",
            "extra bond per atom",
            "extra angle per atom",
            "extra dihedral per atom",
            "extra improper per atom",
            "extra special per atom",
            "ellipsoids",
            "lines",
            "triangles",
            "bodies",
            "xlo xhi",
            "ylo yhi",
            "zlo zhi",
            "xy xz yz",
        ]
        sections_re = "(" + "|".join(sections).replace(" ", "\\s+") + ")"
        header_fields_re = "(" + "|".join(header_fields).replace(" ", "\\s+") + ")"
        # seperate content to image contents
        with open(lmp_file, 'r') as rf:
            lmp_contents = rf.readlines()

        section = None
        header = True
        for idx, ii in tqdm(enumerate(lmp_contents), total=len(lmp_contents), desc="Loading data"):
            if comment is None:
                comment = ii.strip()
                image = Image()
                self.image_list.append(image)
            else:
                ii = re.sub("#.*", "", ii).strip()
                if re.match("^\\s*$", ii):  # skip blank lines
                    continue

            # check for known section names
            if re.match(sections_re, ii):
                section = ii
                header = False
                continue

            if header:
                field = None
                val = None
                _ = re.match("(.*)\\s+" + header_fields_re, ii)
                if _:
                    field = _.group(2).lstrip().rstrip()
                    val = _.group(1).lstrip().rstrip()
                if field is not None and val is not None:
                    if field == "atoms":
                        atom_nums = int(val)
                        image.atom_nums = atom_nums
                    elif field == "xlo xhi":
                        (xlo, xhi) = [float(x) for x in val.split()]
                    elif field == "ylo yhi":
                        (ylo, yhi) = [float(x) for x in val.split()]
                    elif field == "zlo zhi":
                        (zlo, zhi) = [float(x) for x in val.split()]
                    elif field == "xy xz yz":
                        (xy, xz, yz) = [float(x) for x in val.split()]

            if section is not None:
                fields = ii.split()
                if section == "Atoms":  # id *
                    id = int(fields[0])
                    if style == "full" and (len(fields) == 7 or len(fields) == 10):
                        # id mol-id type q x y z [tx ty tz]
                        pos_in[id] = (
                            int(fields[2]),
                            float(fields[4]),
                            float(fields[5]),
                            float(fields[6]),
                        )
                        mol_id_in[id] = int(fields[1])
                        charge_in[id] = float(fields[3])
                        if len(fields) == 10:
                            travel_in[id] = (
                                int(fields[7]),
                                int(fields[8]),
                                int(fields[9]),
                            )
                    elif style == "atomic" and (
                            len(fields) == 5 or len(fields) == 8
                    ):
                        # id type x y z [tx ty tz]
                        pos_in[id] = (
                            int(fields[1]),
                            float(fields[2]),
                            float(fields[3]),
                            float(fields[4]),
                        )
                        if len(fields) == 8:
                            travel_in[id] = (
                                int(fields[5]),
                                int(fields[6]),
                                int(fields[7]),
                            )
                    elif (style in ("angle", "bond", "molecular")
                        ) and (len(fields) == 6 or len(fields) == 9):
                        # id mol-id type x y z [tx ty tz]
                        pos_in[id] = (
                            int(fields[2]),
                            float(fields[3]),
                            float(fields[4]),
                            float(fields[5]),
                        )
                        mol_id_in[id] = int(fields[1])
                        if len(fields) == 9:
                            travel_in[id] = (
                                int(fields[6]),
                                int(fields[7]),
                                int(fields[8]),
                            )
                    elif (style == "charge"
                        and (len(fields) == 6 or len(fields) == 9)):
                        # id type q x y z [tx ty tz]
                        pos_in[id] = (
                            int(fields[1]),
                            float(fields[3]),
                            float(fields[4]),
                            float(fields[5]),
                        )
                        charge_in[id] = float(fields[2])
                        if len(fields) == 9:
                            travel_in[id] = (
                                int(fields[6]),
                                int(fields[7]),
                                int(fields[8]),
                            )
                    else:
                        raise RuntimeError(
                            "Style '{}' not supported or invalid "
                            "number of fields {}"
                            "".format(style, len(fields))
                        )
                elif section == "Velocities":  # id vx vy vz
                    vel_in[int(fields[0])] = (
                        float(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                    )
                elif section == "Masses":
                    mass = float(fields[1])
                    mass_in[int(fields[0])] = mass
                    if self.atom_names is None:
                        for key, value in ELEMENTMASSTABLE.items():
                            if abs(value - mass) < 0.001:
                                atom_names.append(key)
                                break
                elif section == "Bonds":  # id type atom1 atom2
                    bonds_in.append(
                        (int(fields[1]), int(fields[2]), int(fields[3]))
                    )
                elif section == "Angles":  # id type atom1 atom2 atom3
                    angles_in.append(
                        (
                            int(fields[1]),
                            int(fields[2]),
                            int(fields[3]),
                            int(fields[4]),
                        )
                    )
                elif section == "Dihedrals":  # id type atom1 atom2 atom3 atom4
                    dihedrals_in.append(
                        (
                            int(fields[1]),
                            int(fields[2]),
                            int(fields[3]),
                            int(fields[4]),
                            int(fields[5]),
                        )
                    )

        # set lattice
        lattice = np.zeros((3, 3))
        lattice[0, 0] = xhi - xlo
        lattice[1, 1] = yhi - ylo
        lattice[2, 2] = zhi - zlo
        if xy is not None:
            lattice[1, 0] = xy
        if xz is not None:
            lattice[2, 0] = xz
        if yz is not None:
            lattice[2, 1] = yz

        # initialize arrays for per-atom quantities
        positions = np.zeros((atom_nums, 3))
        atom_types_image = np.zeros(atom_nums, dtype=int)
        ids = np.zeros(atom_nums, dtype=int)
        types = np.zeros(atom_nums, dtype=int)
        velocities = np.zeros((atom_nums, 3)) if len(vel_in) > 0 else None
        masses = np.zeros(atom_nums) if len(mass_in) > 0 else None
        mol_id = np.zeros(atom_nums, dtype=int) if len(mol_id_in) > 0 else None
        charge = np.zeros(atom_nums, dtype=float) if len(charge_in) > 0 else None
        travel = np.zeros((atom_nums, 3), dtype=int) if len(travel_in) > 0 else None
        bonds = [''] * atom_nums if len(bonds_in) > 0 else None
        angles = [''] * atom_nums if len(angles_in) > 0 else None
        dihedrals = [''] * atom_nums if len(dihedrals_in) > 0 else None

        ind_of_id = {}
        # copy per-atom quantities from read-in values
        for (i, id) in enumerate(pos_in.keys()):
            # by id
            ind_of_id[id] = i
            if sort_by_id:
                ind = id - 1
            else:
                ind = i
            type = pos_in[id][0]
            positions[ind, :] = [pos_in[id][1], pos_in[id][2], pos_in[id][3]]
            if velocities is not None:
                velocities[ind, :] = [vel_in[id][0], vel_in[id][1], vel_in[id][2]]
            if travel is not None:
                travel[ind] = travel_in[id]
            if mol_id is not None:
                mol_id[ind] = mol_id_in[id]
            if charge is not None:
                charge[ind] = charge_in[id]
            ids[ind] = id
            # by type
            types[ind] = type
            if atom_names is None:
                atom_types_image[ind] = type
            else:
                atom_types_image[ind] = atom_names[type - 1]
            if masses is not None:
                masses[ind] = mass_in[type]

        # convert units
        positions = convert(positions, "distance", units, "ASE")
        lattice = convert(lattice, "distance", units, "ASE")
        if masses is not None:
            masses = convert(masses, "mass", units, "ASE")
        if velocities is not None:
            velocities = convert(velocities, "velocity", units, "ASE")

        sc = Counter(atom_types_image)      # a list sc of (atom_types, count) pairs
        atom_type = list(sc.keys())
        atom_type_num = list(sc.values())
        image.lattice = lattice
        image.position = positions
        image.cartesian = True
        image.atom_type = atom_type
        image.atom_type_num = atom_type_num
        image.atom_types_image = atom_types_image

    