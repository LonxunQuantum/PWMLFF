from pwdata import Config

mvm_path = "/data/home/wuxingxing/datas/pwmat_mlff_workdir/hfo2/nep_ff_1image/mvm_10"

config = Config(format="pwmat/movement", data_path=mvm_path)

image = config.images[0]
image = image._set_cartesian()
print()


# list_neigh, dR_neigh, max_ri, Egroup_weight, Divider, Egroup = find_neighbore(data["AtomTypeMap"], data["Position"], data["Lattice"], data["ImageAtomNum"], data["Ei"], 
#                                                                                       self.img_max_types, self.Rc_type, self.Rm_type, self.m_neigh, self.Rc_M, self.Egroup)
