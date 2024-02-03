import re
from tqdm import tqdm
from image import Image
from calculators.const import elements

class POSCAR(object):
    def __init__(self, poscar_file, pbc = None) -> None:
        self.image_list:list[Image] = []
        self.number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
        self.load_poscar_file(poscar_file)

        assert len(self.image_list) > 0, "No system loaded!"
        self.image_list[0].pbc = pbc

    def get(self):
        return self.image_list

    def load_poscar_file(self, poscar_file):
        # seperate content to image contents
        with open(poscar_file, 'r') as rf:
            poscar_contents = rf.readlines()
        
        for idx, ii in tqdm(enumerate(poscar_contents), total=len(poscar_contents), desc="Loading data"):
            if idx == 0:
                image = Image()
                self.image_list.append(image)
                lattice_info = self.parse_lattice(poscar_contents[idx+2:idx+5])
                image.lattice = lattice_info["lattice"]
                atom_names = poscar_contents[idx+5].split()
                image.atom_type_num = [int(_) for _ in poscar_contents[idx+6].split()]
                image.atom_nums = sum(image.atom_type_num)
            elif "direct" in ii.lower() or "cartesian" in ii.lower():
                image.cartesian = "cartesian" in ii.lower()
                position = self.parse_position(poscar_contents[idx+1:idx+image.atom_nums+1], atom_names, image.atom_type_num)
                image.position = position["position"]
                atom_types_image = position["atom_types_image"]
                image.atom_type = [elements.index(atom_names[_]) for _ in range(len(image.atom_type_num))]
                image.atom_types_image = [elements.index(atom_types_image[_]) for _ in range(image.atom_nums)]
        image.image_nums = len(self.image_list)
        print("Load data %s successfully!" % poscar_file)
    
    def parse_lattice(self, lattice_content):
        lattice1 = [float(_) for _ in self.number_pattern.findall(lattice_content[0])]
        lattice2 = [float(_) for _ in self.number_pattern.findall(lattice_content[1])]
        lattice3 = [float(_) for _ in self.number_pattern.findall(lattice_content[2])]
        lattice = [lattice1[:3], lattice2[:3], lattice3[:3]]
        return {"lattice": lattice}
    
    def parse_position(self, position_content, atom_names, atom_type_num):
        position = []
        atom_types_image = []
        for i in range(0, len(position_content)):
            numbers = self.number_pattern.findall(position_content[i])
            position.append([float(_) for _ in numbers[:3]])
            # if position_content[i].split()[3] in elements:
            #     atom_types_image.append(elements.index(position_content[i].split()[3]))
        
        if len(atom_types_image) == 0:
            for atom_name, num in zip(atom_names, atom_type_num):
                atom_types_image.extend([atom_name] * num)   
        return {"position": position, "atom_types_image": atom_types_image}