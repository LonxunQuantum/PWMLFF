import numpy as np
def write_line_to_file(file_path, line, write_patthen="w"):
    with open(file_path, write_patthen) as wf:
        wf.write(line)
        wf.write('\n')

def write_arrays_to_file(file_path, arrays, write_patthen="w"):
    with open(file_path, write_patthen) as wf:
        for data in arrays:
          if isinstance(data, list):
            line = ' '.join(np.array(data).astype('str'))
          else:
             line = "{}".format(data)
          wf.write(line)
          wf.write('\n') 