#include "op_declare.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("calculate_force", 
          &torch_launch_calculate_force,
          "calculate force kernel warpper");
    
    m.def("calculate_force_grad", 
          &torch_launch_calculate_force_grad,
          "calculate force grad kernel warpper");

    m.def("calculate_virial_force", 
          &torch_launch_calculate_virial_force,
          "calculate virial force kernel warpper");

    m.def("calculate_virial_force_grad", 
          &torch_launch_calculate_virial_force_grad,
          "calculate virial force grad kernel warpper");
}

TORCH_LIBRARY(op, m) {
    m.def("calculate_force", torch_launch_calculate_force);

    m.def("calculate_force_grad", torch_launch_calculate_force_grad);

    m.def("calculate_virial_force", torch_launch_calculate_virial_force);

    m.def("calculate_virial_force_grad", torch_launch_calculate_virial_force_grad);
}