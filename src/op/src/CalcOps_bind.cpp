#include <torch/torch.h>
#include <torch/extension.h>

#include "../include/CalcOps.h"

TORCH_LIBRARY(CalcOps_cuda, m) {
    m.def("calculateForce", calculateForce);
    m.def("calculateVirial", calculateVirial);
    m.def("calculateCompress", calculateCompress);
    m.def("calculateNepFeat", calculateNepFeat);
}