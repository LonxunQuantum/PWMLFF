import os
import numpy as np
import numpy.linalg as LA

def save_to_movement(image_data_all: list, output_path: str, output_file: str, is_cartesian: bool = True):
    output_file = open(os.path.join(output_path, output_file), 'w')
    for i in range(len(image_data_all)):
        image_data = image_data_all[i]
        if is_cartesian:
            image_data.position = np.dot(image_data.position, LA.inv(image_data.lattice))       # cartesian position to fractional position
        # with open(os.path.join(output_path, output_file), 'a') as wf:
        output_file.write(" %d atoms,Iteration (fs) = %16.10E, Etot,Ep,Ek (eV) = %16.10E  %16.10E   %16.10E, SCF = %d\n"\
                            % (image_data.atom_nums, 0.0, image_data.Etot, image_data.Etot, 0.0, image_data.scf))
        output_file.write(" MD_INFO: METHOD(1-VV,2-NH,3-LV,4-LVPR,5-NHRP) TIME(fs) TEMP(K) DESIRED_TEMP(K) AVE_TEMP(K) TIME_INTERVAL(fs) TOT_TEMP(K) \n")
        output_file.write("          *    ************   ********   ********   ********    ********    ********\n")
        output_file.write("     TOTAL MOMENTUM\n")
        output_file.write("     ********    ********    ********\n")
        output_file.write(" MD_VV_INFO: Basic Velocity Verlet Dynamics (NVE), Initialized total energy(Hartree)\n")
        output_file.write("          *******              \n")
        output_file.write("Lattice vector (Angstrom)\n")
        for j in range(3):
            if image_data.stress != []:
                output_file.write("  %16.10E    %16.10E    %16.10E     stress (eV): %16.10E    %16.10E    %16.10E\n" % (image_data.lattice[j][0], image_data.lattice[j][1], image_data.lattice[j][2], image_data.virials[j][0], image_data.virials[j][1], image_data.virials[j][2]))
            else:
                output_file.write("  %16.10E    %16.10E    %16.10E\n" % (image_data.lattice[j][0], image_data.lattice[j][1], image_data.lattice[j][2]))
        output_file.write("  Position (normalized), move_x, move_y, move_z\n")
        for j in range(image_data.atom_nums):
            output_file.write(" %4d    %20.15F    %20.15F    %20.15F    1 1 1\n"\
                                % (image_data.atom_types_image[j], image_data.position[j][0], image_data.position[j][1], image_data.position[j][2]))
        output_file.write("  Force (-force, eV/Angstrom)\n")
        for j in range(image_data.atom_nums):
            output_file.write(" %4d    %20.15F    %20.15F    %20.15F\n"\
                                % (image_data.atom_types_image[j], -image_data.force[j][0], -image_data.force[j][1], -image_data.force[j][2]))
        output_file.write(" -------------------------------------\n")
    output_file.close()
    print("Convert to %s successfully!" % output_file)