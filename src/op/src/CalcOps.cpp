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
    double rcut_radial,
    int64_t multi_feat_num) 
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
        ctx->save_for_backward({coeff2, d12_radial, NL_radial, dfeat_c2, dfeat_2b, dfeat_2b_noc, atom_map});
        ctx->saved_data["multi_feat_num"] = multi_feat_num;

        return {feats};
    }

torch::autograd::variable_list CalculateNepFeat::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        std::cout << "CalculateNepFeat::backward grad_output 2b shape: " << grad_output[0].sizes() << std::endl;
        // auto gradin = grad_output[0].to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();         // 获取数据，假设数据是 float 类型
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         printf("2b grad_output[%ld][%ld][:] = %f %f %f %f %f\n", i, j,
        //          gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + 0],
        //          gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + 1],
        //          gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + 2],
        //          gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + 3],
        //          gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + 4]);
        //         }
        //     printf("\n");
        // }
        auto saved = ctx->get_saved_variables();
        auto coeff2 = saved[0];
        auto d12_radial = saved[1];
        auto NL_radial = saved[2];
        auto dfeat_c2 = saved[3];
        auto dfeat_2b = saved[4];
        auto dfeat_2b_noc = saved[5];
        auto atom_map = saved[6];
        int64_t multi_feat_num = ctx->saved_data["multi_feat_num"].toInt();
        
        auto result = CalculateNepFeatGrad::apply(grad_output[0], coeff2, d12_radial, NL_radial, dfeat_c2, dfeat_2b, dfeat_2b_noc, atom_map, multi_feat_num);
        // return CalculateNepFeatFuncs::backward(grad_output, coeff2, d12_radial, dfeat_c2, dfeat_2b, atom_map);
        auto grad_coeff2 = result[0];
        auto grad_d12_radial = result[1];
        

        // auto gradin = grad_coeff2.to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();         // 获取数据，假设数据是 float 类型
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // int64_t gradin_dim3 = gradin_shape[3];
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         for (int64_t l = 0; l < gradin_dim2; ++l) {
        //             printf("2b grad_coeff2[%ld][%ld][%ld][:] = ", i, j, l);
        //             for (int64_t p = 0; p < gradin_dim3; ++p) {
        //             printf("%f ", gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3
        //                                     + j * gradin_dim2 * gradin_dim3 
        //                                     + l * gradin_dim3 
        //                                     + p]);
        //             }
        //             printf("\n");
        //         }
        //     }
        // }
        // printf("====end====");
        // gradin = grad_d12_radial.to(torch::kCPU);  // 确保它在 CPU 上
        // gradin_data = gradin.data<double>();         // 获取数据，假设数据是 float 类型
        // gradin_shape = gradin.sizes();
        // gradin_dim0 = gradin_shape[0];
        // gradin_dim1 = gradin_shape[1];
        // gradin_dim2 = gradin_shape[2];
        // gradin_dim3 = gradin_shape[3];
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         for (int64_t l = 0; l < gradin_dim2; ++l) {
        //             if (j > 3) continue;
        //             printf("2b grad_d12_radial[%ld][%ld][%ld][:] = ", i, j, l);
        //             for (int64_t p = 0; p < gradin_dim3; ++p) {
        //             printf("%f ", gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3
        //                                     + j * gradin_dim2 * gradin_dim3 
        //                                     + l * gradin_dim3 
        //                                     + p]);
        //             }
        //             printf("\n");
        //         }
        //     }
        // }
        // printf("====end====");

        return {grad_coeff2, 
                grad_d12_radial, 
                torch::autograd::Variable(), 
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
            at::Tensor NL_radial,
            at::Tensor dfeat_c2,
            at::Tensor dfeat_2b,
            at::Tensor dfeat_2b_noc,
            at::Tensor atom_map,
            int64_t multi_feat_num) 
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
        auto grad_coeff2 = torch::zeros({atom_types, atom_types, n_max_2b, n_base_2b}, d12_radial.options());
        auto grad_d12_radial = torch::zeros({batch_size, atom_nums, maxneighs, 4}, d12_radial.options());

        torch_launch_calculate_nepfeat_grad(grad_input, dfeat_c2, dfeat_2b, atom_map, 
                     batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b, atom_types, multi_feat_num, grad_coeff2, grad_d12_radial);
        // grad_coeff2 = grad_coeff2.permute({0, 2, 1, 3});
        ctx->save_for_backward({coeff2, d12_radial, NL_radial, dfeat_c2, dfeat_2b, dfeat_2b_noc, grad_input, atom_map});
        ctx->saved_data["multi_feat_num"] = multi_feat_num;
        return {grad_coeff2, grad_d12_radial};

    }

torch::autograd::variable_list CalculateNepFeatGrad::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_second) 
    {
        printf("\n\n=========== second grad ===========\n");
        auto saved = ctx->get_saved_variables();
        auto coeff2 = saved[0];
        auto d12_radial = saved[1];
        auto NL_radial = saved[2];
        auto dfeat_c2 = saved[3];
        auto dfeat_2b = saved[4];
        auto dfeat_2b_noc = saved[5];
        auto de_feat = saved[6];
        auto atom_map = saved[7];

        auto dims_2b = coeff2.sizes();
        int64_t atom_types = dims_2b[0];
        int64_t n_max_2b = dims_2b[2];
        int64_t n_base_2b = dims_2b[3];

        auto dims_image = d12_radial.sizes();
        int64_t batch_size = dims_image[0];
        int64_t atom_nums = dims_image[1];
        int64_t maxneighs = dims_image[2];

        int64_t multi_feat_num = ctx->saved_data["multi_feat_num"].toInt();
        std::cout << "CalculateNepFeatGrad::backward grad_second[0] shape: " << grad_second[0].sizes() << std::endl;
        std::cout << "CalculateNepFeatGrad::backward grad_second[1] shape: " << grad_second[1].sizes() << std::endl;
        // auto gradin = grad_second[1].to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // int64_t gradin_dim3 = gradin_shape[3];
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         if (j != 1) continue;
        //         for (int64_t k = 0; k < gradin_dim2; ++k) {
        //             printf("2b grad_second[1][%d][%d][%d][:] = %f %f %f %f\n", i, j, k, 
        //             gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3 + j * gradin_dim2 * gradin_dim3 + k * gradin_dim3],
        //             gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3 + j * gradin_dim2 * gradin_dim3 + k * gradin_dim3+1],
        //             gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3 + j * gradin_dim2 * gradin_dim3 + k * gradin_dim3+2],
        //             gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3 + j * gradin_dim2 * gradin_dim3 + k * gradin_dim3+3]);
        //         }
        //     }
        // }
        // printf("\n==================\n");


        // auto gradsecond_gradout = torch::zeros({batch_size, atom_nums, maxneighs, n_max_2b, 4}, coeff2.options());
        // gradsecond_gradout.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
        //                            torch::indexing::Slice(), 0}, dfeat_2b.squeeze(-1));
        // // 3. 逐元素乘法：C * A_expanded，得到 [batch, atom_nums, maxneighs, n_max_2b, 4]
        // gradsecond_gradout = (grad_second[1].unsqueeze(-2) * gradsecond_gradout).sum(-1).sum(-2);  // 对最后一个维度进行求和，得到 [batch, atom_nums, n_max_2b]

        // auto gradin = gradsecond_gradout.to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // printf("gradsecond_gradout[1] shape [%d  %d  %d]\n", gradin_dim0, gradin_dim1, gradin_dim2);
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         printf("gradsecond_gradout[%ld][%ld][:] = ", i, j);
        //         for (int64_t k = 0; k < gradin_dim2; ++k) {
        //             printf("%f ", gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + k]);
        //         }
        //         printf("\n");  // 换行，表示输出一组数据结束
        //     }
        // }
        // printf("\n==================\n");

        // 对gradout的导数
        auto gradsecond_gradout = torch::zeros({batch_size, atom_nums, n_max_2b}, coeff2.options());
        torch_launch_calculate_nepfeat_secondgradout(grad_second[1], dfeat_2b, batch_size, atom_nums, maxneighs, n_max_2b, gradsecond_gradout);

        // 对C的导数
        // auto gradin = dfeat_2b_noc.to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // int64_t gradin_dim3 = gradin_shape[3];
        // int64_t gradin_dim4 = gradin_shape[4];
        // printf("dfeat_2b_noc shape [%d  %d  %d  %d %d]\n", gradin_dim0, gradin_dim1, gradin_dim2, gradin_dim3, gradin_dim4);
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         for (int64_t k = 0; k < gradin_dim2; ++k) {
        //             if ((j == 10) and (k == 10)){
        //                 for (int64_t l = 0; l < gradin_dim3; ++l) {
        //                 printf("dfeat_2b_noc[%ld][%ld][%ld][:] %f %f %f %f \n", i, j, k,
        //                     gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3 * 4  + j * gradin_dim2 * gradin_dim3 * 4 + k * gradin_dim3 * 4 + l * 4],
        //                     gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3 * 4  + j * gradin_dim2 * gradin_dim3 * 4 + k * gradin_dim3 * 4 + l * 4+1],
        //                     gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3 * 4  + j * gradin_dim2 * gradin_dim3 * 4 + k * gradin_dim3 * 4 + l * 4+2],
        //                     gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3 * 4  + j * gradin_dim2 * gradin_dim3 * 4 + k * gradin_dim3 * 4 + l * 4+3]
        //                 );
        //                 }
        //             }
        //         }
        //     }
        // }
        // printf("\n==================\n\n");

        // gradin = grad_second[1].to(torch::kCPU);  // 确保它在 CPU 上
        // gradin_data = gradin.data<double>();
        // gradin_shape = gradin.sizes();
        // gradin_dim0 = gradin_shape[0];
        // gradin_dim1 = gradin_shape[1];
        // gradin_dim2 = gradin_shape[2];
        // gradin_dim3 = gradin_shape[3];
        // printf("grad_second[1] shape [%d  %d  %d  %d]\n", gradin_dim0, gradin_dim1, gradin_dim2, gradin_dim3);
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         for (int64_t k = 0; k < gradin_dim2; ++k) {
        //             if ((j == 10) and (k == 10)){
        //                 printf("grad_second[1][%ld][%ld][%ld][:] %f %f %f %f \n", i, j, k,
        //                     gradin_data[i * gradin_dim1 * gradin_dim2 * 4  + j * gradin_dim2 * 4 + k * 4 + 0],
        //                     gradin_data[i * gradin_dim1 * gradin_dim2 * 4  + j * gradin_dim2 * 4 + k * 4 + 1],
        //                     gradin_data[i * gradin_dim1 * gradin_dim2 * 4  + j * gradin_dim2 * 4 + k * 4 + 2],
        //                     gradin_data[i * gradin_dim1 * gradin_dim2 * 4  + j * gradin_dim2 * 4 + k * 4 + 3]
        //                 );
        //             }
        //         }
        //     }
        // }
        // printf("\n==================\n\n");

        // gradin = de_feat.to(torch::kCPU);  // 确保它在 CPU 上
        // gradin_data = gradin.data<double>();
        // gradin_shape = gradin.sizes();
        // gradin_dim0 = gradin_shape[0];
        // gradin_dim1 = gradin_shape[1];
        // gradin_dim2 = gradin_shape[2];
        // printf("de_feat shape [%d  %d  %d]\n", gradin_dim0, gradin_dim1, gradin_dim2);
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         if (j == 10){
        //             printf("de_feat[%ld][%ld][:]=", gradin_dim0, gradin_dim1);
        //             for (int64_t k = 0; k < gradin_dim2; ++k) {
        //                 printf("%f ",
        //                     gradin_data[i * gradin_dim1 * gradin_dim2 + j * gradin_dim2 + k]
        //                 );
        //             }
        //             printf("\n");
        //         }
        //     }
        // }
        // printf("\n==================\n\n");

        auto gradsecond_c2 = torch::zeros({atom_types, atom_types, n_max_2b, n_base_2b}, coeff2.options());
        torch_launch_calculate_nepfeat_secondgradout_c2(grad_second[1], de_feat, dfeat_2b_noc, atom_map, NL_radial, 
                            batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b, atom_types, multi_feat_num, gradsecond_c2);

        // auto gradsecond_c2 = torch::zeros({atom_types, atom_types, n_max_2b, n_base_2b}, coeff2.options());
        // gradsecond_c2 = de_feat.unsqueeze(2).unsqueeze(3).unsqueeze(5);
        // gradsecond_c2 =  gradsecond_c2 * dfeat_2b_noc.unsqueeze(4);
        // gradsecond_c2 = grad_second[1].unsqueeze(3).unsqueeze(4) * gradsecond_c2;
        // gradsecond_c2 = gradsecond_c2.sum(-1);

        // auto gradin = gradsecond_c2.to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // int64_t gradin_dim3 = gradin_shape[3];
        // int64_t gradin_dim4 = gradin_shape[4];
        // printf("tmpgrad shape [%d  %d  %d  %d  %d]\n", gradin_dim0, gradin_dim1, gradin_dim2, gradin_dim3, gradin_dim4);
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         for (int64_t k = 0; k < gradin_dim2; ++k) {
        //             if (j < 3 and k < 20){
        //                 for (int64_t l = 0; l < gradin_dim3; ++l) {
        //                     printf("tmpgrad[%ld][%ld][%ld][%ld][:] = ", i, j, k, l);
        //                     for (int64_t q = 0; q < gradin_dim4; ++q) {
        //                         int64_t idx = i * gradin_dim1 * gradin_dim2 * gradin_dim3 * gradin_dim4 + 
        //                                                     j * gradin_dim2 * gradin_dim3 * gradin_dim4 + 
        //                                                                   k * gradin_dim3 * gradin_dim4 + 
        //                                                                                 l * gradin_dim4;
        //                         printf("%f ", i, j, k, l, q, gradin_data[idx + q]);
        //                     }
        //                     printf("\n");
        //                     }
        //             }
        //         }
        //     }
        // }
        // printf("\n==================\n\n");

        // gradsecond_c2 = gradsecond_c2.sum(0).sum(0).sum(0).unsqueeze(0).unsqueeze(0);
        // gradsecond_c2 = gradsecond_c2.permute({0, 1, 3, 2});

        // auto gradin = gradsecond_c2.to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // int64_t gradin_dim3 = gradin_shape[3];
        // printf("gradsecond_gradout shape [%d  %d  %d  %d]\n", gradin_dim0, gradin_dim1, gradin_dim2, gradin_dim3);
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         for (int64_t k = 0; k < gradin_dim2; ++k) {
        //             printf("gradsecond_gradout[%ld][%ld][%ld][:] = ", i, j, k);
        //             for (int64_t l = 0; l < gradin_dim3; ++l) {
        //                 printf("%f ", gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3  + j * gradin_dim2 * gradin_dim3 + k * gradin_dim3 + l]);
        //             }
        //             printf("\n");  // 换行，表示输出一组数据结束
        //         }
        //     }
        // }
        // printf("\n==================\n");

        return {
            gradsecond_gradout,
            gradsecond_c2,
            torch::autograd::Variable(),
            torch::autograd::Variable(), 
            torch::autograd::Variable(), 
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable()
            };
    }

torch::autograd::variable_list calculateNepFeat(
    at::Tensor coeff2,
    at::Tensor d12_radial,
    at::Tensor NL_radial,
    at::Tensor atom_map,
    at::Tensor feats,
    double rcut_radial,
    int64_t feat_multi_nums) 
    {
        return CalculateNepFeat::apply(coeff2, d12_radial, NL_radial, atom_map, feats, rcut_radial, feat_multi_nums);
    }


torch::autograd::variable_list CalculateNepMbFeat::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor coeff3,
    at::Tensor d12,
    at::Tensor NL,
    at::Tensor atom_map,
    at::Tensor feats,
    int64_t feat_2b_num,
    int64_t lmax_3,
    int64_t lmax_4,
    int64_t lmax_5,
    double rcut_angular) 
    {
        auto dims = coeff3.sizes();
        int64_t atom_types = dims[0];
        int64_t n_max = dims[2];
        int64_t n_base = dims[3];
        auto dims_image = d12.sizes();
        int64_t batch_size = dims_image[0];
        int64_t atom_nums = dims_image[1];
        int64_t maxneighs = dims_image[2];
        // options() 集成dtype和device
        const int64_t NUM_OF_ABC = 24;
        auto dfeat_c3 = torch::zeros({batch_size, atom_nums, atom_types, n_base}, d12.options());
        auto dfeat_3b = torch::zeros({batch_size, atom_nums, maxneighs, n_max}, d12.options()); // [i, j, n] dfeature/drij
        auto dfeat_3b_noc = torch::zeros({batch_size, atom_nums, maxneighs, n_base, 4}, d12.options()); // [i, j, n] dfeature/drij without c
        auto sum_fxyz = torch::zeros({batch_size, atom_nums, n_max * NUM_OF_ABC}, d12.options());
        std::cout << "CalculateNepMbFeat:forward atom_map shape: " << atom_map.sizes() << std::endl;
        std::cout << "CalculateNepMbFeat:forward NL shape: " << NL.sizes() << std::endl;
        std::cout << "CalculateNepMbFeat:forward d12 shape: " << d12.sizes() << std::endl;
        torch_launch_calculate_nepmbfeat(coeff3, d12, NL, atom_map,
                                feats, dfeat_c3, dfeat_3b, dfeat_3b_noc, sum_fxyz,
                                    rcut_angular, batch_size, atom_nums, maxneighs, n_max, n_base, lmax_3, lmax_4, lmax_5, atom_types);
                     // 交换coeff2维度，方便在核函数中检索c下标
        ctx->save_for_backward({coeff3, d12, NL, dfeat_c3, dfeat_3b, dfeat_3b_noc, sum_fxyz, atom_map});
        ctx->saved_data["rcut_angular"] = rcut_angular;
        ctx->saved_data["feat_2b_num"] = feat_2b_num;
        ctx->saved_data["lmax_3"] = lmax_3;
        ctx->saved_data["lmax_4"] = lmax_4;
        ctx->saved_data["lmax_5"] = lmax_5;
        return {feats};
    }

torch::autograd::variable_list CalculateNepMbFeat::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto coeff3 = saved[0];
        auto d12 = saved[1];
        auto NL = saved[2];
        auto dfeat_c3 = saved[3];
        auto dfeat_3b = saved[4];
        auto dfeat_3b_noc = saved[5];
        auto sum_fxyz = saved[6];
        auto atom_map = saved[7];
        double rcut_angular = ctx->saved_data["rcut_angular"].toDouble();
        int64_t feat_2b_num = ctx->saved_data["feat_2b_num"].toInt();
        int64_t lmax_3 = ctx->saved_data["lmax_3"].toInt();
        int64_t lmax_4 = ctx->saved_data["lmax_4"].toInt();
        int64_t lmax_5 = ctx->saved_data["lmax_5"].toInt();
        auto result = CalculateNepMbFeatGrad::apply(grad_output[0], coeff3, d12, NL, dfeat_c3, dfeat_3b, dfeat_3b_noc, sum_fxyz, atom_map, feat_2b_num, lmax_3, lmax_4, lmax_5, rcut_angular);
        auto grad_coeff3 = result[0];
        auto grad_d12_angular = result[1];
        return {grad_coeff3, 
                grad_d12_angular, 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable()
                }; // f2的梯度 coeff的梯度 因为不需要，所以不计算，只占位
    }

torch::autograd::variable_list CalculateNepMbFeatGrad::forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor grad_input,
            at::Tensor coeff3,
            at::Tensor d12,
            at::Tensor NL,
            at::Tensor dfeat_c3,
            at::Tensor dfeat_3b,
            at::Tensor dfeat_3b_noc,
            at::Tensor sum_fxyz,
            at::Tensor atom_map,
            int64_t feat_2b_num,
            int64_t lmax_3,
            int64_t lmax_4,
            int64_t lmax_5,
            double rcut_angular) 
    {
        printf("\n\n=========== first grad mb ===========\n\n");
        auto dims = coeff3.sizes();
        int64_t atom_types = dims[0];
        int64_t n_max = dims[2];
        int64_t n_base = dims[3];

        auto dims_image = d12.sizes();
        int64_t batch_size = dims_image[0];
        int64_t atom_nums = dims_image[1];
        int64_t maxneighs = dims_image[2];

        // std::cout << "CalculateNepMbFeatGrad::forward grad_input shape: " << grad_input.sizes() << std::endl;
        // auto gradin = grad_input.to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();         // 获取数据，假设数据是 float 类型
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         for (int64_t l = 0; l < 4; ++l) {
        //             printf("3b grad_input[%ld][%ld][%ld][:] = ", i, j, l);
        //             for (int64_t k = 0; k < n_max; ++k) {
        //                 printf("%f ", gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + l * n_max + k]);
        //             }
        //             printf("\n");
        //         }
        //     }
        // }
        // printf("\n==================\n");

        int64_t feat_3b_num = 0;
        int nlm = 0;
        if (lmax_3 > 0) {
            feat_3b_num += n_max * lmax_3;
            nlm += lmax_3;
        }
        if (lmax_4 > 0) {
            feat_3b_num += n_max;
            nlm += 1;
        } 
        if (lmax_5 > 0) {
            feat_3b_num += n_max;
            nlm += 1;
        }
        // options() 集成dtype和device
        const int64_t NUM_OF_ABC = 24;
        auto grad_coeff3= torch::zeros({atom_types, atom_types, n_max, n_base}, d12.options());
        auto grad_d12_angular = torch::zeros({batch_size*atom_nums, maxneighs, 4}, d12.options());
        auto dfeat_dc   = torch::zeros({batch_size*atom_nums, atom_types, n_max, n_base}, d12.options());
        auto dfeat_drij = torch::zeros({batch_size*atom_nums, maxneighs, feat_3b_num, 4}, d12.options());
        torch_launch_calculate_nepmbfeat_grad(
            grad_input.view({batch_size*atom_nums, feat_3b_num}), 
                coeff3, d12, NL, atom_map, rcut_angular, 
                    batch_size, atom_nums, maxneighs, feat_2b_num, n_max, n_base, 
                        lmax_3, lmax_4, lmax_5, atom_types,
                            sum_fxyz, grad_coeff3, grad_d12_angular, dfeat_dc, dfeat_drij);

        ctx->save_for_backward({coeff3, d12, NL, sum_fxyz, 
                            dfeat_drij.view({batch_size, atom_nums, maxneighs, feat_3b_num, 4}), grad_input, atom_map});
                            
        ctx->saved_data["rcut_angular"] = rcut_angular;
        ctx->saved_data["feat_2b_num"] = feat_2b_num; //for multi body feature index offset in grad_input (dEi/dfeat)
        ctx->saved_data["lmax_3"] = lmax_3;
        ctx->saved_data["lmax_4"] = lmax_4;
        ctx->saved_data["lmax_5"] = lmax_5;
        return {grad_coeff3, grad_d12_angular.view({batch_size, atom_nums, maxneighs, 4})};
    }

torch::autograd::variable_list CalculateNepMbFeatGrad::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_second) 
    {
        printf("\n\n=========== second grad mb ===========\n");
        std::cout << "CalculateNepMbFeatGrad::backward grad_second[0] shape: " << grad_second[0].sizes() << std::endl;
        std::cout << "CalculateNepMbFeatGrad::backward grad_second[1] shape: " << grad_second[1].sizes() << std::endl;

        auto saved = ctx->get_saved_variables();
        auto coeff3 = saved[0];
        auto d12 = saved[1];
        auto NL = saved[2];
        auto sum_fxyz = saved[3];
        auto dfeat_drij = saved[4];
        auto de_feat = saved[5];
        auto atom_map = saved[6];
        double rcut_angular = ctx->saved_data["rcut_angular"].toDouble();
        int64_t feat_2b_num = ctx->saved_data["feat_2b_num"].toInt();
        int64_t lmax_3 = ctx->saved_data["lmax_3"].toInt();
        int64_t lmax_4 = ctx->saved_data["lmax_4"].toInt();
        int64_t lmax_5 = ctx->saved_data["lmax_5"].toInt();

        auto dims = coeff3.sizes();
        int64_t atom_types = dims[0];
        int64_t n_max = dims[2];
        int64_t n_base = dims[3];

        auto dims_image = d12.sizes();
        int64_t batch_size = dims_image[0];
        int64_t atom_nums = dims_image[1];
        int64_t maxneighs = dims_image[2];

        int64_t feat_3b_num = 0;
        if (lmax_3 > 0) feat_3b_num += n_max * lmax_3;
        if (lmax_4 > 0) feat_3b_num += n_max;
        if (lmax_5 > 0) feat_3b_num += n_max;

        auto gradsecond_gradout = torch::zeros({batch_size, atom_nums, feat_3b_num}, coeff3.options());
        // gradsecond_gradout = (grad_second[1].unsqueeze(-2) * dfeat_drij).sum(-1).sum(-2); //Similar in function to the lower kernel function
        torch_launch_calculate_nepmbfeat_secondgradout(grad_second[1], 
                                                        dfeat_drij, 
                                                        batch_size, 
                                                        atom_nums, 
                                                        maxneighs, 
                                                        feat_3b_num, 
                                                        gradsecond_gradout);

        auto gradsecond_c3 = torch::zeros({atom_types, atom_types, n_max, n_base}, coeff3.options());
        // torch_launch_calculate_nepmbfeat_secondgradout_c3(grad_second[1],
        //                                                     d12,
        //                                                     NL,
        //                                                     de_feat,
        //                                                     sum_fxyz,
        //                                                     atom_map,
        //                                                     rcut_angular,
        //                                                     batch_size, 
        //                                                     atom_nums, 
        //                                                     maxneighs, 
        //                                                     n_max,
        //                                                     n_base, 
        //                                                     atom_types, 
        //                                                     lmax_3, 
        //                                                     lmax_4, 
        //                                                     lmax_5, 
        //                                                     feat_2b_num,
        //                                                     multi_feat_num, 
        //                                                     gradsecond_c3);

        return {
            gradsecond_gradout,
            gradsecond_c3,
            torch::autograd::Variable(),
            torch::autograd::Variable(), 
            torch::autograd::Variable(), 
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(), 
            torch::autograd::Variable(), 
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable()
            };
    }

torch::autograd::variable_list calculateNepMbFeat(
    at::Tensor coeff,
    at::Tensor d12,
    at::Tensor NL,
    at::Tensor atom_map,
    at::Tensor feats,
    int64_t feat_2b_num,
    int64_t lmax_3,
    int64_t lmax_4,
    int64_t lmax_5,
    double rcut) 
    {
        return CalculateNepMbFeat::apply(coeff, d12, NL, atom_map, feats, feat_2b_num, lmax_3, lmax_4, lmax_5, rcut);
    }


//abandon
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
