#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "descriptor.h"

namespace py = pybind11;

PYBIND11_MODULE(descriptor_pybind, m)
{
    py::class_<Descriptor>(m, "Descriptor")
        .def(py::init<>())
        .def("get_feat", &Descriptor::get_feat)
        .def("get_nfeat", &Descriptor::get_nfeat)
        .def("get_neighbor_list", &Descriptor::get_neighbor_list)
        .def("get_dfeat", &Descriptor::get_dfeat)
        .def("get_dfeat2c", &Descriptor::get_dfeat2c)
        .def("get_ddfeat2c", &Descriptor::get_ddfeat2c)
        .def("show", &Descriptor::show);

    py::class_<MultiDescriptor>(m, "MultiDescriptor")
        .def(py::init([](int a, int b, int c, int d, float e, float f, int g, int h, int i, py::array_t<int> j, py::array_t<int> k, py::array_t<int> l, py::array_t<double> m, py::object n)
                      {
            if (n.is_none()) {
                return new MultiDescriptor(a, b, c, d, e, f, g, h, i, j.mutable_data(), k.mutable_data(), l.mutable_data(), m.mutable_data());
            } else {
                py::array_t<double> n_array = n.cast<py::array_t<double>>();
                return new MultiDescriptor(a, b, c, d, e, f, g, h, i, j.mutable_data(), k.mutable_data(), l.mutable_data(), m.mutable_data(), n_array.mutable_data());
            } }))
        // .def("get_feat", [](MultiDescriptor &instance)
        //      {
        //     // Call the original get_feat method
        //     double* result = instance.get_feat();
        //     // Calculate the size of the result
        //     int size = instance.images * instance.natoms * instance.get_nfeat();
        //     // Create a Python array with the result
        //     return py::array_t<double>(size, result); })
        .def("get_feat", [](MultiDescriptor &instance)
             { return py::array_t<double>({instance.images * instance.natoms, instance.get_nfeat()}, instance.get_feat()); })
        .def("get_dfeat", [](MultiDescriptor &instance)
             { return py::array_t<double>({instance.images, instance.natoms, instance.get_nfeat(), instance.max_neighbors, 3}, instance.get_dfeat()); })
        .def("get_dfeat2c", [](MultiDescriptor &instance)
             { return py::array_t<double>({instance.images, instance.natoms, instance.get_nfeat(), instance.ntypes, instance.m1, instance.beta}, instance.get_dfeat2c()); })
        .def("get_ddfeat2c", [](MultiDescriptor &instance)
             { return py::array_t<double>({instance.images, instance.natoms, instance.get_nfeat(), instance.ntypes, instance.m1, instance.beta, instance.max_neighbors, 3}, instance.get_ddfeat2c()); })
        .def("get_neighbor_list", [](MultiDescriptor &instance)
             { return py::array_t<int>({instance.images, instance.natoms, instance.max_neighbors}, instance.get_neighbor_list()); })
        .def("show", &MultiDescriptor::show);

    py::class_<MultiNeighborList>(m, "MultiNeighborList")
        .def(py::init([](int a, float b, int c, int d, int e, py::array_t<int> f, py::array_t<double> g, py::array_t<double> h)
                      { return new MultiNeighborList(a, b, c, d, e, f.mutable_data(), g.mutable_data(), h.mutable_data()); }))
        .def("show", &MultiNeighborList::show)
        .def("get_num_neigh_all", [](MultiNeighborList &m)
             { return py::array_t<int>({m.images, m.natoms, m.ntypes}, m.get_num_neigh_all()); })
        .def("get_neighbors_list_all", [](MultiNeighborList &m)
             { return py::array_t<int>({m.images, m.natoms, m.ntypes, m.max_neighbors}, m.get_neighbors_list_all()); })
        .def("get_dR_neigh_all", [](MultiNeighborList &m)
             { return py::array_t<double>({m.images, m.natoms, m.ntypes, m.max_neighbors, 4}, m.get_dR_neigh_all()); });
}