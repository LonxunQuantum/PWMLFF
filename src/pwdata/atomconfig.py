import re
from tqdm import tqdm
from collections import Counter
from image import Image

class CONFIG(object):
    def __init__(self, config_file, pbc = None) -> None:
        self.image_list:list[Image] = []
        self.number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
        self.load_config_file(config_file)

        assert len(self.image_list) > 0, "No system loaded!"
        self.image_list[0].pbc = pbc

    def get(self):
        return self.image_list
    
    def load_config_file(self, config_file):
        # seperate content to image contents
        with open(config_file, 'r') as rf:
            config_contents = rf.readlines()
        
        for idx, ii in tqdm(enumerate(config_contents), total=len(config_contents), desc="Loading data"):
            if "lattice" in ii.lower():
                atom_nums = int(config_contents[0].split()[0])
                lattice_info = self.parse_lattice(config_contents[idx+1:idx+4])
                image = Image(**lattice_info)
                self.image_list.append(image)
                image.atom_nums = atom_nums
            elif "position" in ii.lower():
                position = self.parse_position(config_contents[idx+1:idx+image.atom_nums+1])
                image.position = position["position"]
                image.atom_type = position["atom_type"]
                image.atom_type_num = position["atom_type_num"]
                image.atom_types_image = position["atom_types_image"]
        image.image_nums = len(self.image_list)
        print("Load data %s successfully!" % config_file)
    
    def parse_lattice(self, lattice_content):
        lattice1 = [float(_) for _ in self.number_pattern.findall(lattice_content[0])]
        lattice2 = [float(_) for _ in self.number_pattern.findall(lattice_content[1])]
        lattice3 = [float(_) for _ in self.number_pattern.findall(lattice_content[2])]
        lattice = [lattice1[:3], lattice2[:3], lattice3[:3]]
        return {"lattice": lattice}
    
    def parse_position(self, position_content):
        atom_types_image = []
        position = []
        for i in range(0, len(position_content)):
            numbers = self.number_pattern.findall(position_content[i])
            atom_types_image.append(int(numbers[0]))
            position.append([float(_) for _ in numbers[1:4]])
        counter = Counter(atom_types_image)
        atom_type = list(counter.keys())
        atom_type_num = list(counter.values())
        assert sum(atom_type_num) == len(position)
        return {"atom_type": atom_type, "atom_type_num": atom_type_num, "atom_types_image": atom_types_image, "position": position}