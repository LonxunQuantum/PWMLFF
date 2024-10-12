'''
description: 
    get value of param in json_input which is required parameters which need input by user
        if the parameter is not specified in json_input, raise error and print error log to user.
param {str} param
param {dict} json_input
param {str} info 
return {*}
author: wuxingxing
'''
def get_required_parameter(param:str, json_input:dict, detail_infos:str=None):
    if param not in json_input.keys():
        error_info = "Input error! : The {} parameter is missing and must be specified in input json file!".format(param)
        if detail_infos is not None:
            error_info += "\n{}\n".format(detail_infos)
        raise Exception(error_info)
    return json_input[param]

'''
description: 
    get value of param in json_input,
        if the parameter is not specified in json_input, return the default parameter value 'default_value'
param {str} param
param {dict} json_input
param {*} default_value
return {*}
author: wuxingxing
'''
def get_parameter(param:str, json_input:dict, default_value, format_type:str=None):
    res = None
    if param not in json_input.keys():
        res = default_value
    else:
        res = json_input[param]
    
    if format_type is not None:
        if format_type.upper() == "upper".upper():
            res = res.upper()
    return res


def convert_keys_to_lowercase(dictionary:dict):
    if isinstance(dictionary, dict):
        new_dict = {}
        for key, value in dictionary.items():
            new_key = key.lower()
            new_value = convert_keys_to_lowercase(value)  # 递归处理嵌套的字典
            new_dict[new_key] = new_value
        return new_dict
    elif isinstance(dictionary, list):
        return [convert_keys_to_lowercase(item) for item in dictionary]
    else:
        return dictionary