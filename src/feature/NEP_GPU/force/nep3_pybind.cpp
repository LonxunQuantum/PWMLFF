#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "nep3.cuh"

namespace py = pybind11;

PYBIND11_MODULE(nep3_module, m) {
    py::class_<NEP3>(m, "NEP3")
        .def(py::init<>())  // 暴露默认构造函数
        .def("init_from_file", &NEP3::init_from_file, 
             py::arg("file_potential"), 
             py::arg("is_rank_0"), 
             py::arg("in_device_id"))
        .def("compute_pwmlff", [](NEP3& self, 
                                  int N, int NM, 
                                  py::array_t<int> itype_cpu, 
                                  py::array_t<double> box_cpu, 
                                  py::array_t<double> position_cpu, 
                                  py::array_t<double> cpu_potential_per_atom, 
                                  py::array_t<double> cpu_force_per_atom, 
                                  py::array_t<double> cpu_total_virial) {
            // 获取 NumPy 数组的指针
            auto itype_ptr = itype_cpu.mutable_data();
            auto box_ptr = box_cpu.mutable_data();
            auto position_ptr = position_cpu.mutable_data();
            auto potential_ptr = cpu_potential_per_atom.mutable_data();
            auto force_ptr = cpu_force_per_atom.mutable_data();
            auto virial_ptr = cpu_total_virial.mutable_data();

            // 调用 NEP3 的 compute_pwmlff 方法
            self.compute_pwmlff(N, NM, itype_ptr, box_ptr, position_ptr, potential_ptr, force_ptr, virial_ptr);
        }, 
        py::arg("N"), 
        py::arg("NM"), 
        py::arg("itype_cpu"), 
        py::arg("box_cpu"), 
        py::arg("position_cpu"), 
        py::arg("cpu_potential_per_atom"), 
        py::arg("cpu_force_per_atom"), 
        py::arg("cpu_total_virial"));
}

