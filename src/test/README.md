# pwmlff_auto_test
This project is used for automated testing of PWMLFF and LAMMPS interfaces

# 操作命令，src/test 在当前目录下执行
python auto_test.py  example/template_train.json

# 修改环境信息
在 "envs" 中增加或者修改自己的环境信息即可

# 增加测试案例

在数据目录 `"path_prefix"` 下准备测试案例的json文件，如下例子（测试数据是raw_files），以及对应的训练 json 文件，在测试时，会将`测试数据` 以及设置的 `epoch` 自动写入json文件中，并使用该json文件中的参数做训练，训练完毕后，会自动将测试数据执行 `PWMLFF test ` 命令做推理。

在 `"train_inputs"` 中增加或者修改对应字典即可，每一条 dict 对应一个测试案例。

# 执行需要的测试案例

在 `"work_list"` 中配置即可。

 "train_inputs" 这里设置需要测试的测试案例。
 "envs" 和 "epochs" 配合使用， envs 列出了需要测试的代码环境，epochs 列出该测试代码环境下测试例子的epoch。


 例如这里"envs":[5, 6], "epochs":[1, 1], "train_inputs":[0, 1, 2, 3, 4, 5]，表示：
 在envs 的下标为5和6的环境下，分别测试测试案例0，1，2，3，4，5，并且在5和6的环境下，分别设置这些案例的epoch为1，2。

# 测试案例的json文件格式

```txt
    "path_prefix": "../../example",
    "train_inputs": [
        {"_idx":0, "json_file" : "Cu/train.json", "model_type" :"DP", "do_test": true, "raw_files":["Cu/0_300_MOVEMENT","Cu/1_500_MOVEMENT"], "format":"pwmat/movement"},
        {"_idx":1, "json_file" : "EC/train.json",  "model_type" :"NN", "do_test": true, "raw_files":["EC/EC_MOVEMENT"], "format":"pwmat/movement"},
        {"_idx":2, "json_file" : "LiGePS/train.json", "model_type" :"DP", "do_test": true, "raw_files":["LiGePS/100_1200k_movement"], "format":"pwmat/movement"},
        {"_idx":3, "json_file" : "LiSi/train.json",  "model_type" :"NEP", "do_test": true, "raw_files":["LiSi/MOVEMENT"], "format":"pwmat/movement"},
        {"_idx":4, "json_file" : "NEP/train.json", "model_type" :"NN", "do_test": true, "raw_files":["NEP/mvms/mvm_init_000_50", "NEP/mvms/mvm_init_001_50", "NEP/mvms/mvm_init_002_50"], "format":"pwmat/movement"},
        {"_idx":5, "json_file" : "SiC/train.json",  "model_type" :"LINEAR", "do_test": true, "raw_files":["SiC/1_300_MOVEMENT", "SiC/2_300_MOVEMENT"], "format":"pwmat/movement"}
    ],

    "work_dir": "/data/home/wuxingxing/datas/auto_test_pwmlff/tmp_work/train_pwmlff2",
    "work_list": [
        {
            "_work_type":"mcloud envs(gpu and cpu) for DP, NEP",
            "envs":[5, 6],
            "epochs":[1, 2],
            "train_inputs":[0, 1, 2, 3, 4, 5]
        }
    ]
```