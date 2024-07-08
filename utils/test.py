# from pwdata import Config


# list_neigh, dR_neigh, max_ri, Egroup_weight, Divider, Egroup = find_neighbore(data["AtomTypeMap"], data["Position"], data["Lattice"], data["ImageAtomNum"], data["Ei"], 
#                                                                                       self.img_max_types, self.Rc_type, self.Rm_type, self.m_neigh, self.Rc_M, self.Egroup)


# import torch
# import glob
# import os

def test():
    pass

if __name__ == "__main__":
    # work_dir = "/data/home/wuxingxing/datas/PWMLFF_library_data"
    # model_dirs = glob.glob(os.path.join(work_dir, "*/models/*/checkpoint.pth.tar"))
    # rights = []
    # errors = []
    # for model_path in model_dirs:
    #     m = torch.load(model_path, map_location=torch.device('cpu'))
    #     layers = m['state_dict'].keys()
    #     has_resnet = False
    #     for layer in layers:
    #         if "resnet" in layer:
    #             has_resnet = True
    #             rights.append(model_path)
    #             break
    #     if has_resnet is False:
    #         errors.append(model_path)
    
    # print("Right path:")
    # print(rights)
    # print("error paths:")
    # print(errors)

    # print("sum: {} are right, {} are error!".format(len(rights) , len(errors)))
    # from pwdata import Config
    # import numpy as np
    # image = Config(format="pwmat/movement", data_path="/data/home/wuxingxing/codespace/PWMLFF_nep/pwmat_mlff_workdir/hfo2/nep_ff_1image/mvm_10")
    # m = image.images[0]
    # position = np.array(m.position).reshape(-1, 3)
    # lattice = np.array(m.lattice).reshape(3, 3)
    # np.dot(position, lattice)
    # print()

    # import numpy as np
    # pynep = "/data/home/wuxingxing/datas/pwmat_mlff_workdir/hfo2/gpumd2lkf/nep.txt"
    # nep  = "/data/home/wuxingxing/datas/pwmat_mlff_workdir/hfo2/gpumd2lkf/model_record/nep.txt"
    # with open(pynep, 'r') as rf:
    #     py_line = rf.readlines()
    # with open(nep, "r") as rf:
    #     _line = rf.readlines()

    # for i in range(7, len(_line)):
    #     a = float(py_line[i])
    #     b = float(_line[i])
    #     if abs(a-b) > 0.001:
    #         print(i, a, b)


    # import glob
    # import os
    # dirs = ["/data/home/wuxingxing/datas/PWMLFF_library_data/nep-data/16_metal/pwdata_1", "/data/home/wuxingxing/datas/PWMLFF_library_data/nep-data/16_metal/pwdata_2"]
    # res = []
    # for d in dirs:
    #     dir_list = os.listdir(d)
    #     for _dir in dir_list:
    #         if not os.path.isdir(os.path.join(d, _dir)): 
    #             continue
    #         sub_dir = os.listdir(os.path.join(d, _dir))
    #         res.append(os.path.join(d, _dir,sub_dir[0]))
    # content = ""
    # for _ in res:
    #     line = "\"{}\",".format(_)
    #     print(line)
    #     content += "{}\n".format(line)
    # with open("/data/home/wuxingxing/datas/PWMLFF_library_data/nep-data/16_metal/train/tmp/dir_list", 'w') as wf:
    #     wf.writelines(content)
    
    res = [23296, 11520, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1, 2560, 80, 80, 1]

    param_nums = []
    param_sum = 0
    block_size = 2721

    for index, param in enumerate(res):
        param_num = param
        print(param_num, param_sum, block_size)
        if param_sum + param_num > block_size:
            #when the first layer params nums more than block_size, 
            # the first P matrix will be [], this 'elif' is handling for this bug
            if param_sum == 0:
                param_sum = param_num
                continue
            param_nums.append(param_sum)
            param_sum = param_num
        else:
            param_sum += param_num
    
    param_nums.append(param_sum)
    print(param_nums)

