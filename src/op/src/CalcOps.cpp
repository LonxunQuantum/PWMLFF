#include <torch/torch.h>
#include <iostream>
#include "../include/CalcOps.h"
#include "../include/op_declare.h"

torch::autograd::variable_list CalculateForceFuncs::forward(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        int nghost = nghost_tensor.item<int>();
        auto force = torch::zeros({batch_size, natoms + nghost, 3}, dE.options());
        force.slice(1, 0, natoms) = F;
        torch_launch_calculate_force(list_neigh, dE, Ri_d, batch_size, natoms, neigh_num, force, nghost);
        return {force};
        
    }

torch::autograd::variable_list CalculateForceFuncs::backward(
    torch::autograd::variable_list grad_output,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        auto grad = torch::zeros_like(dE);
        torch_launch_calculate_force_grad(list_neigh, Ri_d, grad_output[0], batch_size, natoms, neigh_num, grad);
        return {torch::autograd::Variable(), grad, torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
    }

torch::autograd::variable_list CalculateForce::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor) 
    {
        ctx->save_for_backward({list_neigh, dE, Ri_d});
        return CalculateForceFuncs::forward(list_neigh, dE, Ri_d, F, nghost_tensor);
    }

torch::autograd::variable_list CalculateForce::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto list_neigh = saved[0];
        auto dE = saved[1];
        auto Ri_d = saved[2];
        return CalculateForceFuncs::backward(grad_output, list_neigh, dE, Ri_d);
    }
    
torch::autograd::variable_list calculateForce(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor) 
    {
        return CalculateForce::apply(list_neigh, dE, Ri_d, F, nghost_tensor);
    }

// the following is the code virial
torch::autograd::variable_list CalculateVirialFuncs::forward(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        int nghost = nghost_tensor.item<int>();
        auto virial = torch::zeros({batch_size, 9}, dE.options());
        auto atom_virial = torch::zeros({batch_size, natoms + nghost, 9}, dE.options());
        torch_launch_calculate_virial_force(list_neigh, dE, Rij, Ri_d, batch_size, natoms, neigh_num, virial, atom_virial, nghost);
        return {virial, atom_virial};
    }

torch::autograd::variable_list CalculateVirialFuncs::backward(
    torch::autograd::variable_list grad_output,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        auto grad = torch::zeros_like(dE);
        torch_launch_calculate_virial_force_grad(list_neigh, Rij, Ri_d, grad_output[0], batch_size, natoms, neigh_num, grad);
        return {torch::autograd::Variable(), grad, torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
    }

torch::autograd::variable_list CalculateVirial::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor) 
    {
        ctx->save_for_backward({list_neigh, dE, Rij, Ri_d});
        return CalculateVirialFuncs::forward(list_neigh, dE, Rij, Ri_d, nghost_tensor);
    }

torch::autograd::variable_list CalculateVirial::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto list_neigh = saved[0];
        auto dE = saved[1];
        auto Rij = saved[2];
        auto Ri_d = saved[3];
        return CalculateVirialFuncs::backward(grad_output, list_neigh, dE, Rij, Ri_d);
    }

torch::autograd::variable_list calculateVirial(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor) 
    {
        return CalculateVirial::apply(list_neigh, dE, Rij, Ri_d, nghost_tensor);
    }
    
// the following is the code compress
torch::autograd::variable_list CalculateCompressFuncs::forward(
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        auto dims = coefficient.sizes();
        int64_t sij_num = dims[0];
        int64_t layer_node = dims[1];
        int64_t coe_num = dims[2];
        auto G = torch::empty({sij_num, layer_node}, f2.options());
        torch_launch_calculate_compress(f2, coefficient, sij_num, layer_node, coe_num, G);
        return {G};
    }

torch::autograd::variable_list CalculateCompressFuncs::backward(
    torch::autograd::variable_list grad_output,
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        auto dims = coefficient.sizes();
        int64_t sij_num = dims[0];
        int64_t layer_node = dims[1];
        int64_t coe_num = dims[2];
        auto grad = torch::zeros({sij_num, layer_node}, f2.options());
        torch_launch_calculate_compress_grad(f2, coefficient, grad_output[0], sij_num, layer_node, coe_num, grad);
        auto grad_out = torch::sum(grad_output[0]*grad, -1).unsqueeze(-1); //对应坐标位置相乘，然后算加法
        return {grad_out, torch::autograd::Variable()}; // f2的梯度 coeff的梯度 因为不需要，所以不计算，只占位
    }

torch::autograd::variable_list CalculateCompress::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        ctx->save_for_backward({f2, coefficient});
        return CalculateCompressFuncs::forward(f2, coefficient);
    }

torch::autograd::variable_list CalculateCompress::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto f2 = saved[0];
        auto coefficient = saved[1];
        return CalculateCompressFuncs::backward(grad_output, f2, coefficient);
    }

torch::autograd::variable_list calculateCompress(
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        return CalculateCompress::apply(f2, coefficient);
    }

torch::autograd::variable_list CalculateNepFeat::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor coeff2,
    at::Tensor d12_radial,
    at::Tensor NL_radial,
    at::Tensor atom_map,
    at::Tensor feats,
    double rcut_radial) 
    {
        auto dims_2b = coeff2.sizes();
        int64_t atom_types = dims_2b[0];
        int64_t n_max_2b = dims_2b[2];
        int64_t n_base_2b = dims_2b[3];

        auto dims_image = d12_radial.sizes();
        int64_t batch_size = dims_image[0];
        int64_t atom_nums = dims_image[1];
        int64_t maxneighs = dims_image[2];
        // options() 集成dtype和device
        // auto feat_2b  = torch::zeros({batch_size, atom_nums, n_max_2b}, d12_radial.options());
        auto dfeat_c2 = torch::zeros({batch_size, atom_nums, atom_types, n_base_2b}, d12_radial.options());
        auto dfeat_2b = torch::zeros({batch_size, atom_nums, maxneighs, n_max_2b}, d12_radial.options()); // [i, j, n] dfeature/drij
        auto dfeat_2b_noc = torch::zeros({batch_size, atom_nums, maxneighs, n_base_2b, 4}, d12_radial.options()); // [i, j, n] dfeature/drij without c
        
        torch_launch_calculate_nepfeat(coeff2, d12_radial, NL_radial, atom_map, 
                                rcut_radial, feats, dfeat_c2, dfeat_2b, dfeat_2b_noc,
                                    batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b, atom_types);
                     // 交换coeff2维度，方便在核函数中检索c下标
        ctx->save_for_backward({coeff2, d12_radial, dfeat_c2, dfeat_2b, dfeat_2b_noc, atom_map});
        return {feats};
    }

torch::autograd::variable_list CalculateNepFeat::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto coeff2 = saved[0];
        auto d12_radial = saved[1];
        auto dfeat_c2 = saved[2];
        auto dfeat_2b = saved[3];
        auto dfeat_2b_noc = saved[4];
        auto atom_map = saved[5];
        auto result = CalculateNepFeatGrad::apply(grad_output[0], coeff2, d12_radial, dfeat_c2, dfeat_2b, dfeat_2b_noc, atom_map);
        // return CalculateNepFeatFuncs::backward(grad_output, coeff2, d12_radial, dfeat_c2, dfeat_2b, atom_map);
        auto grad_coeff2 = result[0];
        auto grad_d12_radial = result[1];
        return {grad_coeff2, 
                grad_d12_radial, 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable()
                }; // f2的梯度 coeff的梯度 因为不需要，所以不计算，只占位
    }

torch::autograd::variable_list CalculateNepFeatGrad::forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor grad_input,
            at::Tensor coeff2,
            at::Tensor d12_radial,
            at::Tensor dfeat_c2,
            at::Tensor dfeat_2b,
            at::Tensor dfeat_2b_noc,
            at::Tensor atom_map) 
    {
        printf("\n\n=========== first grad ===========\n\n");
        std::cout << "CalculateNepFeatGrad::forward grad_input shape: " << grad_input.sizes() << std::endl;
        auto gradin = grad_input.to(torch::kCPU);  // 确保它在 CPU 上
        auto gradin_data = gradin.data<double>();         // 获取数据，假设数据是 float 类型
        auto gradin_shape = gradin.sizes();
        int64_t gradin_dim0 = gradin_shape[0];
        int64_t gradin_dim1 = gradin_shape[1];
        int64_t gradin_dim2 = gradin_shape[2];
        // 遍历前面三维
        for (int64_t i = 0; i < gradin_dim0; ++i) {
            for (int64_t j = 0; j < gradin_dim1; ++j) {
                printf("grad_input[%ld][%ld][:] = ", i, j);
                for (int64_t k = 0; k < gradin_dim2; ++k) {
                    printf("%f ", gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + k]);
                }
                printf("\n");  // 换行，表示输出一组数据结束
            }
        }
        printf("\n==================\n");

        auto dims_2b = coeff2.sizes();
        int64_t atom_types = dims_2b[0];
        int64_t n_max_2b = dims_2b[2];
        int64_t n_base_2b = dims_2b[3];

        auto dims_image = d12_radial.sizes();
        int64_t batch_size = dims_image[0];
        int64_t atom_nums = dims_image[1];
        int64_t maxneighs = dims_image[2];
        // options() 集成dtype和device
        auto grad_coeff2 = torch::zeros({atom_types, n_max_2b, atom_types, n_base_2b}, d12_radial.options());
        auto grad_d12_radial = torch::zeros({batch_size, atom_nums, maxneighs, 4}, d12_radial.options());
        // printf("back grad_output[0,0,:] %f %f %f %f %f\n", grad_output[0][0,0,0], grad_output[0][0,0,1], grad_output[0][0,0,2], grad_output[0][0,0,3], grad_output[0][0,0,4]);
        
        torch_launch_calculate_nepfeat_grad(grad_input, dfeat_c2, dfeat_2b, atom_map, 
                     batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b, atom_types, grad_coeff2, grad_d12_radial);
        grad_coeff2 = grad_coeff2.permute({0, 2, 1, 3});
        // return CalculateNepFeatFuncs::backward(grad_output, coeff2, d12_radial, dfeat_c2, dfeat_2b, atom_map);
        // feats = results[0];
        // auto dfeat_c2 = results[1];
        // auto dfeat_2b = results[2];
        // 输出 grad_second[1] 的数据

        auto grad1 = grad_coeff2.to(torch::kCPU);  // 确保它在 CPU 上
        auto grad1_data = grad1.data<double>();         // 获取数据，假设数据是 float 类型
        auto grad1_shape = grad1.sizes();
        int64_t grad1_dim0 = grad1_shape[0];
        int64_t grad1_dim1 = grad1_shape[1];
        int64_t grad1_dim2 = grad1_shape[2];
        int64_t grad1_dim3 = grad1_shape[3];
        printf("grad_coeff2 shape is: [%ld, %ld, %ld, %ld]\n", grad1_dim0, grad1_dim1, grad1_dim2, grad1_dim3);

        // 遍历前面三维
        for (int64_t i = 0; i < grad1_dim0; ++i) {
            if (i > 0) continue;
            for (int64_t j = 0; j < grad1_dim1; ++j) {
                if (j > 0) continue;
                for (int64_t k = 0; k < grad1_dim2; ++k) {
                    printf("grad_coeff2[%ld][%ld][%ld][:] = ", i, j, k);
                    for (int64_t l = 0; l < grad1_dim3; ++l) {
                        printf("%f ", grad1_data[i * grad1_dim1 * grad1_dim2 * grad1_dim3 + j * grad1_dim2 * grad1_dim3 + k * grad1_dim3 + l]);
                    }
                    printf("\n");  // 换行，表示输出一组数据结束
                }
            }
        }

        printf("\n==================\n\n\n");
        ctx->save_for_backward({coeff2, d12_radial, dfeat_c2, dfeat_2b, dfeat_2b_noc, grad_input, atom_map});
        // return {feats};
        return {grad_coeff2, grad_d12_radial};

    }

torch::autograd::variable_list CalculateNepFeatGrad::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_second) 
    {
        printf("\n\n=========== second grad ===========\n");
        // 输出 grad_second[0] 的数据
        auto grad0 = grad_second[0].to(torch::kCPU);  // 确保它在 CPU 上
        auto grad0_data = grad0.data<double>();         // 获取数据，假设数据是 float 类型
        auto grad0_shape = grad0.sizes();
        int64_t grad0_dim0 = grad0_shape[0];
        int64_t grad0_dim1 = grad0_shape[1];
        int64_t grad0_dim2 = grad0_shape[2];
        int64_t grad0_dim3 = grad0_shape[3];
        printf("grad_second[0] shape is: [%ld, %ld, %ld, %ld]\n", grad0_dim0, grad0_dim1, grad0_dim2, grad0_dim3);

        // 遍历前面三维
        // for (int64_t i = 0; i < grad0_dim0; ++i) {
        //     for (int64_t j = 0; j < grad0_dim1; ++j) {
        //         for (int64_t k = 0; k < grad0_dim2; ++k) {
        //             printf("grad_second[0][%ld][%ld][%ld][:] = ", i, j, k);
        //             for (int64_t l = 0; l < grad0_dim3; ++l) {
        //                 printf("%f ", grad0_data[i * grad0_dim1 * grad0_dim2 * grad0_dim3 + j * grad0_dim2 * grad0_dim3 + k * grad0_dim3 + l]);
        //             }
        //             printf("\n");  // 换行，表示输出一组数据结束
        //         }
        //     }
        // }

        // 输出 grad_second[1] 的数据
        auto grad1 = grad_second[1].to(torch::kCPU);  // 确保它在 CPU 上
        auto grad1_data = grad1.data<double>();         // 获取数据，假设数据是 float 类型
        auto grad1_shape = grad1.sizes();
        int64_t grad1_dim0 = grad1_shape[0];
        int64_t grad1_dim1 = grad1_shape[1];
        int64_t grad1_dim2 = grad1_shape[2];
        int64_t grad1_dim3 = grad1_shape[3];
        printf("grad_second[1] shape is: [%ld, %ld, %ld, %ld]\n", grad1_dim0, grad1_dim1, grad1_dim2, grad1_dim3);

        // 遍历前面三维
        for (int64_t i = 0; i < grad1_dim0; ++i) {
            if (i > 0) continue;
            for (int64_t j = 0; j < grad1_dim1; ++j) {
                if (j > 0) continue;
                for (int64_t k = 0; k < grad1_dim2; ++k) {
                    printf("grad_second[1][%ld][%ld][%ld][:] = ", i, j, k);
                    for (int64_t l = 0; l < grad1_dim3; ++l) {
                        printf("%f ", grad1_data[i * grad1_dim1 * grad1_dim2 * grad1_dim3 + j * grad1_dim2 * grad1_dim3 + k * grad1_dim3 + l]);
                    }
                    printf("\n");  // 换行，表示输出一组数据结束
                }
            }
        }
        auto saved = ctx->get_saved_variables();
        auto coeff2 = saved[0];
        auto d12_radial = saved[1];
        auto dfeat_c2 = saved[2];
        auto dfeat_2b = saved[3];
        auto dfeat_2b_noc = saved[4];
        auto de_feat = saved[5];
        auto atom_map = saved[6];

        auto dims_2b = coeff2.sizes();
        int64_t atom_types = dims_2b[0];
        int64_t n_max_2b = dims_2b[2];
        int64_t n_base_2b = dims_2b[3];

        auto dims_image = d12_radial.sizes();
        int64_t batch_size = dims_image[0];
        int64_t atom_nums = dims_image[1];
        int64_t maxneighs = dims_image[2];

        torch::Tensor de_dc = torch::tensor({
                {{{-3.085065640374e+00,  2.613270752172e-01, -2.551048042619e+00,
                        -3.295589837978e+00,  1.286051728045e+00, -6.190445205078e-01,
                        -6.742115649906e+00,  1.050187007124e+00,  4.222530528202e+00,
                        -1.090248413718e+01,  7.889501161148e-01,  5.480331851499e+00,
                        -8.180466890807e+00},
                        { 3.638631757922e+01,  1.120293311037e+01,  9.119602221880e+00,
                            2.448150436873e+01,  2.412819679054e+01,  1.724837440149e+01,
                            3.258917302293e-02,  3.650603100754e+01,  2.924998361996e+01,
                        -1.613174873272e+01,  3.600260937347e+01,  3.669386395865e+01,
                        -7.712037709560e+00},
                        { 2.843407717065e+01,  9.505409510241e+00,  6.015163367021e+00,
                            1.839583762032e+01,  2.092031168828e+01,  1.395919133804e+01,
                        -4.124846968013e+00,  3.115722675790e+01,  2.692274525336e+01,
                        -2.025078813816e+01,  3.069999840862e+01,  3.388874465059e+01,
                        -1.164885375622e+01},
                        {-6.032194262562e+00, -3.090234842347e+00,  2.992526858974e-01,
                        -2.818494673781e+00, -7.399840639448e+00, -3.688346653611e+00,
                            6.816614077259e+00, -1.026976104712e+01, -1.160373113029e+01,
                            1.506096920929e+01, -9.829568305278e+00, -1.478001536469e+01,
                            1.019834056019e+01},
                        {-9.283176462712e+00,  2.525635746362e+00, -1.022565721072e+01,
                        -1.168570709935e+01,  8.693726320171e+00, -7.011663364537e-01,
                        -2.995337122048e+01,  9.197691419690e+00,  2.215605089007e+01,
                        -5.028475354226e+01,  7.927074577701e+00,  2.863992848157e+01,
                        -3.719948291831e+01}}}
            });
        de_dc = de_dc.to(coeff2.device());
        std::cout << "de_dc shape: " << de_dc.sizes() << std::endl;

        // 下面的没问题了，都对
        auto gradsecond_gradout = torch::zeros({batch_size, atom_nums, maxneighs, n_max_2b, 4}, coeff2.options());
        std::cout << "\n\ngradsecond A 8 dimensions: " << dfeat_2b.sizes() << std::endl;
        std::cout << "\n\ngradsecond C 9 dimensions: " << grad_second[1].sizes() << std::endl;
        gradsecond_gradout.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
                                   torch::indexing::Slice(), 0}, dfeat_2b.squeeze(-1));
        // 3. 逐元素乘法：C * A_expanded，得到 [batch, atom_nums, maxneighs, n_max_2b, 4]
        gradsecond_gradout = (grad_second[1].unsqueeze(-2) * gradsecond_gradout).sum(-1).sum(-2);  // 对最后一个维度进行求和，得到 [batch, atom_nums, n_max_2b]

        // 打印结果的维度
        std::cout << "Result shape: " << gradsecond_gradout.sizes() << std::endl;

        
        // 对C的导数
        auto gradsecond_c = torch::zeros({atom_types, atom_types, n_max_2b, n_base_2b}, coeff2.options());
        
        // auto de_dc_4 = torch::zeros({atom_types, atom_types, n_max_2b, n_base_2b, 4}, coeff2.options());
        // de_dc_4.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
        //                            torch::indexing::Slice(), 0}, de_dc.squeeze(-1));
        // std::cout << "de_dc_4 shape: " << de_dc_4.sizes() << std::endl;

        // gradsecond_c = grad_second[1].sum(0).sum(0).sum(0);
        // std::cout << "gradsecond_c 0 shape: " << gradsecond_c.sizes() << std::endl;
        // gradsecond_c = gradsecond_c * de_dc_4;
        // gradsecond_c = gradsecond_c.sum(-1);
        // std::cout << "\n\ngradsecond_c 1 dimensions: " << gradsecond_c.sizes() << std::endl;
        
        gradsecond_c = de_feat.unsqueeze(2).unsqueeze(3).unsqueeze(5);
        std::cout << "\n\ngradsecond_c 0 dimensions: " << gradsecond_c.sizes() << std::endl;
        std::cout << "\n\ndfeat_2b_noc 0 dimensions: " << dfeat_2b_noc.unsqueeze(4).sizes() << std::endl;
        gradsecond_c =  gradsecond_c * dfeat_2b_noc.unsqueeze(4);
        std::cout << "\n\ngradsecond_c 1 dimensions: " << gradsecond_c.sizes() << std::endl;
        gradsecond_c = grad_second[1].unsqueeze(3).unsqueeze(4) * gradsecond_c;
        std::cout << "\n\ngradsecond_c 2 dimensions: " << gradsecond_c.sizes() << std::endl;
        gradsecond_c = gradsecond_c.sum(-1).sum(0).sum(0).sum(0).unsqueeze(0).unsqueeze(0);
        std::cout << "\n\ngradsecond_c 3 dimensions: " << gradsecond_c.sizes() << std::endl;
        gradsecond_c = gradsecond_c.permute({0, 1, 3, 2});

        // gradsecond_c = grad_second[1].unsqueeze(-2) * dfeat_2b_noc;//[batch, atom_nums, maxneighs, 4]* [batch, atom_nums, maxneighs, n_base_2b, 4]->[batch, atom_nums, maxneighs, n_base_2b, 4]
        // std::cout << "\n\ngradsecond_c 2 dimensions: " << gradsecond_c.sizes() << std::endl;

        // gradsecond_c = gradsecond_c.sum(-1);//[batch_size, atom_nums, maxneighs, n_base_2b]
        // std::cout << "\n\ngradsecond_c 3 dimensions: " << gradsecond_c.sizes() << std::endl;

        // gradsecond_c = gradsecond_c.sum(-2);//[batch_size, atom_nums, n_base_2b]
        // std::cout << "\n\ngradsecond_c 4 dimensions: " << gradsecond_c.sizes() << std::endl;

        // gradsecond_c = gradsecond_c.sum(-3);//[atom_nums, n_base_2b]
        // std::cout << "\n\ngradsecond_c 5 dimensions: " << gradsecond_c.sizes() << std::endl;

        // gradsecond_c = gradsecond_c.sum(0);//[n_base_2b]
        // std::cout << "\n\ngradsecond_c 6 dimensions: " << gradsecond_c.sizes() << std::endl;

        // gradsecond_c = gradsecond_c.unsqueeze(0).expand({n_max_2b, n_base_2b}).unsqueeze(0).unsqueeze(0);
    
        // std::cout << "\n\ngradsecond_c 7 dimensions: " << gradsecond_c.sizes() << std::endl;

        return {
            gradsecond_gradout,
            gradsecond_c,
            torch::autograd::Variable(),
            torch::autograd::Variable(), 
            torch::autograd::Variable(), 
            torch::autograd::Variable(),
            torch::autograd::Variable()
            };
        // return CalculateNepFeatFuncs::backward(grad_output, coeff2, d12_radial, dfeat_c2, dfeat_2b, atom_map);
    }

torch::autograd::variable_list calculateNepFeat(
    at::Tensor coeff2,
    at::Tensor d12_radial,
    at::Tensor NL_radial,
    at::Tensor atom_map,
    at::Tensor feats,
    double rcut_radial) 
    {
        return CalculateNepFeat::apply(coeff2, d12_radial, NL_radial, atom_map, feats, rcut_radial);
    }

//multi feature of nep
torch::autograd::variable_list CalculateNepFeatmbFuncs::forward(
            at::Tensor coeff2,
            at::Tensor coeff3,
            at::Tensor d12,
            at::Tensor d12_3b,
            at::Tensor NL,
            at::Tensor atom_map,
            double rcut_radial,
            double rcut_angular,
            int64_t lmax_3,
            int64_t lmax_4,
            int64_t lmax_5
            ) 
    {
        auto dims_2b = coeff2.sizes();
        int64_t atom_types = dims_2b[0];
        int64_t n_max_2b = dims_2b[2];
        int64_t n_base_2b = dims_2b[3];

        auto dims_3b = coeff3.sizes();
        int64_t n_max_3b = dims_3b[2];
        int64_t n_base_3b = dims_3b[3];

        auto dims_image = d12.sizes();
        int64_t batch_size = dims_image[0];
        int64_t atom_nums = dims_image[1];
        int64_t maxneighs = dims_image[2];

        int64_t feat_2b_num = 0;
        int64_t feat_3b_num = 0;
        feat_2b_num = n_max_2b;
        if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
        if (lmax_4 > 0) feat_3b_num += n_max_3b;
        if (lmax_5 > 0) feat_3b_num += n_max_3b;

        const int64_t NUM_OF_ABC = 24;
        auto feats = torch::zeros({batch_size*atom_nums, feat_2b_num+feat_3b_num}, d12.options());
        auto sum_fxyz = torch::zeros({batch_size*atom_nums, n_max_3b * NUM_OF_ABC}, d12.options());
        
        torch_launch_calculate_nepfeatmb(coeff2, coeff3, 
                    d12.view({batch_size*atom_nums, maxneighs, 4}), 
                        NL.view({batch_size*atom_nums, maxneighs}), 
                            atom_map.repeat(batch_size), 
                                rcut_radial, rcut_angular, 
                                    feats, sum_fxyz, batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b, n_max_3b, n_base_3b, 
                                        lmax_3, lmax_4, lmax_5, atom_types);
                     // 交换coeff2维度，方便在核函数中检索c下标
        return {feats.reshape({batch_size, atom_nums, feat_2b_num+feat_3b_num}), sum_fxyz};
    }

torch::autograd::variable_list CalculateNepFeatmbFuncs::backward(
            torch::autograd::variable_list grad_output,
            at::Tensor coeff2,
            at::Tensor coeff3,
            at::Tensor d12,
            at::Tensor d12_3b,
            at::Tensor NL,
            at::Tensor atom_map,
            at::Tensor sum_fxyz,
            double rcut_radial,
            double rcut_angular,
            int64_t lmax_3,
            int64_t lmax_4,
            int64_t lmax_5) 
    {
        auto dims_2b = coeff2.sizes();
        int64_t atom_types = dims_2b[0];
        int64_t n_max_2b = dims_2b[2];
        int64_t n_base_2b = dims_2b[3];

        auto dims_3b = coeff3.sizes();
        int64_t n_max_3b = dims_3b[2];
        int64_t n_base_3b = dims_3b[3];

        auto dims_image = d12.sizes();
        int64_t batch_size = dims_image[0];
        int64_t atom_nums = dims_image[1];
        int64_t maxneighs = dims_image[2];

        int64_t feat_2b_num = 0;
        int64_t feat_3b_num = 0;

        feat_2b_num = n_max_2b;
        if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
        if (lmax_4 > 0) feat_3b_num += n_max_3b;
        if (lmax_5 > 0) feat_3b_num += n_max_3b;

        auto grad_coeff2 = torch::zeros({atom_types, atom_types, n_max_2b, n_base_2b}, d12.options());
        auto grad_d12_radial = torch::zeros({batch_size*atom_nums, maxneighs, 4}, d12.options());
        auto grad_coeff3 = torch::zeros({atom_types, atom_types, n_max_3b, n_base_3b}, d12.options());
        auto grad_d12_3b = torch::zeros({batch_size*atom_nums, maxneighs, 4}, d12.options());
        
        torch_launch_calculate_nepfeatmb_grad(grad_output[0].view({batch_size*atom_nums, feat_2b_num+feat_3b_num}), 
                                            coeff2, coeff3, d12, NL, atom_map, rcut_radial, rcut_angular, 
                                                batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b, n_max_3b, n_base_3b, 
                                                    lmax_3, lmax_4, lmax_5, atom_types,
                                                        sum_fxyz, grad_coeff2, grad_d12_radial, grad_coeff3, grad_d12_3b);
        // grad_coeff2 = grad_coeff2.permute({0, 2, 1, 3});
        // grad_coeff3 = grad_coeff3.permute({0, 2, 1, 3});
        return {grad_coeff2, grad_coeff3, 
            grad_d12_radial.view({batch_size, atom_nums, maxneighs, 4}), 
                grad_d12_3b.view({batch_size, atom_nums, maxneighs, 4}),
                    torch::autograd::Variable(), torch::autograd::Variable(), 
                        torch::autograd::Variable(), torch::autograd::Variable(), 
                            torch::autograd::Variable(),  torch::autograd::Variable(),
                                torch::autograd::Variable()}; // f2的梯度 coeff的梯度 因为不需要，所以不计算，只占位
    }

torch::autograd::variable_list CalculateNepFeatmb::forward(
    torch::autograd::AutogradContext *ctx,
            at::Tensor coeff2,
            at::Tensor coeff3,
            at::Tensor d12,
            at::Tensor d12_3b,
            at::Tensor NL,
            at::Tensor atom_map,
            double rcut_radial,
            double rcut_angular,
            int64_t lmax_3,
            int64_t lmax_4,
            int64_t lmax_5) 
    {
        auto results = CalculateNepFeatmbFuncs::forward(coeff2, coeff3, d12, d12_3b, NL, atom_map, rcut_radial, rcut_angular, lmax_3, lmax_4, lmax_5);
        auto feats = results[0];
        auto sum_fxyz = results[1];
        ctx->save_for_backward({coeff2, coeff3, d12, d12_3b, NL, atom_map, sum_fxyz});
        //rcut_radial, rcut_angular, lmax_3, lmax_4, lmax_5
        ctx->saved_data["rcut_radial"] = rcut_radial;
        ctx->saved_data["rcut_angular"] = rcut_angular;
        ctx->saved_data["lmax_3"] = lmax_3;
        ctx->saved_data["lmax_4"] = lmax_4;
        ctx->saved_data["lmax_5"] = lmax_5;
        return {feats};
    }

torch::autograd::variable_list CalculateNepFeatmb::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto coeff2 = saved[0];
        auto coeff3 = saved[1];
        auto d12    = saved[2];
        auto d12_3b = saved[3];
        auto NL     = saved[4];
        auto atom_map = saved[5];
        auto sum_fxyz = saved[6];
        double rcut_radial = ctx->saved_data["rcut_radial"].toDouble();
        double rcut_angular = ctx->saved_data["rcut_angular"].toDouble();
        int64_t lmax_3 = ctx->saved_data["lmax_3"].toInt();
        int64_t lmax_4 = ctx->saved_data["lmax_4"].toInt();
        int64_t lmax_5 = ctx->saved_data["lmax_5"].toInt();
        return CalculateNepFeatmbFuncs::backward(grad_output, coeff2, coeff3, d12, d12_3b, NL, atom_map, sum_fxyz, 
                                            rcut_radial, rcut_angular, lmax_3, lmax_4, lmax_5);
    }

torch::autograd::variable_list calculateNepFeatmb(
            at::Tensor coeff2,
            at::Tensor coeff3,
            at::Tensor d12,
            at::Tensor d12_3b,
            at::Tensor NL,
            at::Tensor atom_map,
            double rcut_radial,
            double rcut_angular,
            int64_t lmax_3,
            int64_t lmax_4,
            int64_t lmax_5) 
    {
        return CalculateNepFeatmb::apply(coeff2, coeff3, d12, d12_3b, NL, atom_map, rcut_radial, rcut_angular, lmax_3, lmax_4, lmax_5);
    }
