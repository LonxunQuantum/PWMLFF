import re
import numpy as np
from tqdm import tqdm
from collections import Counter
from image import Image, elements_to_order
# from calculators.const import elements
from lmps import l2Box
from calculators.unitconvert_lmps import convert

class DUMP(object):
    def __init__(self, dump_file, atom_names: list[str] = None) -> None:
        self.image_list:list[Image] = []
        self.number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
        self.atom_names = atom_names
        self.load_dump_file(dump_file)        # load dump file, Adapted from ase: http://wiki.fysik.dtu.dk/ase

        assert len(self.image_list) > 0, "No system loaded!"

    def get(self):
        return self.image_list
    
    def load_dump_file(self, dump_file):
        # seperate content to image contents
        with open(dump_file, 'r') as rf:
            dump_contents = rf.readlines()
        
        # avoid references before assignment in case of incorrect file structure
        lattice, lattice_disp, pbc = None, None, False
        for idx, ii in tqdm(enumerate(dump_contents), total=len(dump_contents), desc="Loading data"):
            if "ITEM: TIMESTEP" in ii:
                image = Image()
                self.image_list.append(image)
            elif "ITEM: NUMBER OF ATOMS" in ii:
                atom_nums = int(dump_contents[idx+1])
                image.atom_nums = atom_nums
            elif "ITEM: BOX BOUNDS" in ii:
            # save labels behind "ITEM: BOX BOUNDS" in triclinic case
                tilt_items = ii.split()[3:]
                lmps_box = dump_contents[idx+1:idx+4]
                diagdisp = np.zeros((3, 2))     # cell dimension convoluted with the displacement vector
                offdiag = np.zeros([3])         # off-diagonal cell elements
                for i in range(3):
                    info = [float(_) for _ in lmps_box[i].split()]
                    diagdisp[i][0] = info[0]
                    diagdisp[i][1] = info[1]
                    if "xy xz yz" in ii:
                        offdiag[i] = info[2]
                    # lmps_box[i] = [float(_) for _ in self.number_pattern.findall(lmps_box[i])]
                lattice, lattice_disp = l2Box(diagdisp, offdiag)

                # Handle pbc conditions
                if len(tilt_items) == 3:
                    pbc_items = tilt_items
                elif len(tilt_items) > 3:
                    pbc_items = tilt_items[3:6]
                else:
                    pbc_items = ["f", "f", "f"]
                pbc = ["p" in d.lower() for d in pbc_items]

            elif "ITEM: ATOMS" in ii:
                colnames = ii.split()[2:]
                traj = [jj.split() for jj in dump_contents[idx+1:idx+image.atom_nums+1]]
                # for jj in dump_contents[idx+1:idx+image.atom_nums+1]:
                info = self.lammps_data_to_config(np.array(traj), colnames, lattice, lattice_disp, atom_nums, self.atom_names)
                image.lattice = info["lattice"]
                # image.lattice_disp = info["lattice_disp"]
                image.atom_type = info["atom_type"]
                image.atom_type_num = info["atom_type_num"]
                image.atom_types_image = info["atom_types_image"]
                image.position = info["positions"]
                image.force = info["forces"]
                image.pbc = pbc
                image.cartesian = True

    def lammps_data_to_config(self,
                              data,
                              colnames,
                              lattice,
                              lattice_disp,
                              atom_nums,
                              atom_names, 
                              order=True,
                              specorder=None,
                              units="metal"):
        """
        Convert LAMMPS data to configuration.               

        Args:
            data (numpy.ndarray): The dump data.
            colnames (list): The column index of the dump data.
            lattice (numpy.ndarray): The lattice vectors.
            lattice_disp (numpy.ndarray): The lattice displacements. (origin shift)
            atom_nums (int): The number of atoms in the lattice.
            atom_names (list): The names of the atoms.
            order (bool, optional): Whether to order the data (Sort atoms by id). Might be faster to turn off. Defaults to True.
                                    Disregarded in case `id` column is not given in file.
            specorder (list, optional): list of species to map lammps types to atom species. Defaults to None.
                                    (usually .dump files to not contain type to species mapping)
            units (str, optional): The units of the data. Defaults to "metal". Now only support "metal", adpat from ASE format.

        Returns:
            dict: The converted configuration.
        """

        # read IDs if given and order if needed
        if "id" in colnames:
            ids = data[:, colnames.index("id")].astype(int)
            if order:
                sort_order = np.argsort(ids)
                data = data[sort_order, :]
            assert max(ids) == atom_nums, "Number of atoms in dump file does not match the number of atoms in the lattice!"
        # determine the elements
        if "element" in colnames:
            # priority to elements written in file
            elements = data[:, colnames.index("element")]
        elif "type" in colnames:
            # fall back to `types` otherwise
            elements = data[:, colnames.index("type")].astype(int)

            # reconstruct types from given specorder
            if specorder:
                elements = [specorder[t - 1] for t in elements]
        else:
            # todo: what if specorder give but no types?
            # in principle the masses could work for atoms, but that needs
            # lots of cases and new code I guess
            raise ValueError("Cannot determine atom types form LAMMPS dump file")
        # elements2type = np.array(atom_names)[elements - 1]
        atom_types_image = elements_to_order(atom_names, elements, atom_nums)
        sc = Counter(atom_types_image)      # a list sc of (atom_types, count) pairs
        atom_type = list(sc.keys())
        atom_type_num = list(sc.values())

        def get_quantity(labels, quantity=None):
            try:
                cols = [colnames.index(label) for label in labels]
                if quantity:
                    return convert(data[:, cols].astype(float), quantity,
                                units, "ASE")

                return data[:, cols].astype(float)
            except ValueError:
                return None

        # Positions
        positions = None
        scaled_positions = None
        if "x" in colnames:
            # doc: x, y, z = unscaled atom coordinates
            positions = get_quantity(["x", "y", "z"], "distance")
        elif "xs" in colnames:
            # doc: xs,ys,zs = scaled atom coordinates
            scaled_positions = get_quantity(["xs", "ys", "zs"])
        elif "xu" in colnames:
            # doc: xu,yu,zu = unwrapped atom coordinates
            positions = get_quantity(["xu", "yu", "zu"], "distance")
        elif "xsu" in colnames:
            # xsu,ysu,zsu = scaled unwrapped atom coordinates
            scaled_positions = get_quantity(["xsu", "ysu", "zsu"])
        else:
            raise ValueError("No atomic positions found in LAMMPS output")

        velocities = get_quantity(["vx", "vy", "vz"], "velocity")
        charges = get_quantity(["q"], "charge")
        forces = get_quantity(["fx", "fy", "fz"], "force")
        # !TODO: how need quaternions be converted?
        quaternions = get_quantity(["c_q[1]", "c_q[2]", "c_q[3]", "c_q[4]"])

        # convert lattice
        lattice = convert(lattice, "distance", units, "ASE")
        lattice_disp = convert(lattice_disp, "distance", units, "ASE")

        all = {}
        # all["elem2type"] = elements2type
        all["atom_types_image"] = atom_types_image
        all["atom_type"] = atom_type
        all["atom_type_num"] = atom_type_num
        if positions is not None:
            all["positions"] = positions
        if scaled_positions is not None:
            all["scaled_positions"] = scaled_positions
        if velocities is not None:
            all["velocities"] = velocities
        if charges is not None:
            all["charges"] = charges
        if forces is not None:
            all["forces"] = forces
        if quaternions is not None:
            all["quaternions"] = quaternions
        if lattice is not None:
            all["lattice"] = lattice
        if lattice_disp is not None:
            all["lattice_disp"] = lattice_disp
    
        return all
    