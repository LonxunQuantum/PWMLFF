import os
from image import frac2cart
from const import elements

def save_to_extxyz(image_data_all: object, image_nums: int, output_path: str, output_file: str):
    output_file = open(os.path.join(output_path, output_file), 'w')
    for i in range(image_nums):
        image_data = image_data_all.image_list[i]
        image_data.position = frac2cart(image_data.position, image_data.lattice)
        output_file.write("%d\n" % image_data.atom_nums)
        # output_file.write("Iteration: %s\n" % image_data.iteration)
        output_head = 'Lattice="%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" Properties=species:S:1:pos:R:3:force:R:3:local_energy:R:1 pbc="T T T"\n'
        output_extended = (image_data.lattice[0][0], image_data.lattice[0][1], image_data.lattice[0][2], 
                                image_data.lattice[1][0], image_data.lattice[1][1], image_data.lattice[1][2], 
                                image_data.lattice[2][0], image_data.lattice[2][1], image_data.lattice[2][2])
        output_file.write(output_head % output_extended)
        for j in range(image_data.atom_nums):
            properties_format = "%s %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f\n"
            properties = (elements[image_data.atom_types_image[j]], image_data.position[j][0], image_data.position[j][1], image_data.position[j][2], 
                            image_data.force[j][0], image_data.force[j][1], image_data.force[j][2], 
                            image_data.atomic_energy[j])
            output_file.write(properties_format % properties)
    output_file.close()
    print("Convert to %s successfully!" % output_file)