import torch
from torch.optim.optimizer import Optimizer
import math


class LKFOptimizer(Optimizer):
    def __init__(
        self,
        params,
        kalman_lambda=0.98,
        kalman_nue=0.9987,
        block_size=5120,
        q0 = None,
        qmin = None,
        p0_weight = None,
        kalman_lambda_weight=None
    ):

        defaults = dict(
            lr=0.1,
            kalman_nue=kalman_nue,
            block_size=block_size,
        )
        
        super(LKFOptimizer, self).__init__(params, defaults)

        self._params = self.param_groups[0]["params"]   

        if len(self.param_groups) != 1 or len(self._params) == 0:
            raise ValueError(
                "LKF doesn't support per-parameter options " "(parameter groups)"
            )

        # NOTE: LKF has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        self._state = self.state[self._params[0]]
        self._state.setdefault("kalman_lambda", kalman_lambda)
        
        self.q0 = q0
        self.qmin = qmin
        self.p0_weight = p0_weight
        self.kalman_lambda_weight = kalman_lambda_weight
        self.__init_P()

    def __init_P(self, q0=None):

        param_nums = []
        param_sum = 0
        block_size = self.__get_blocksize()
        self.data_type = self._params[0].dtype
        self.device = self._params[0].device

        for param_group in self.param_groups:
            params = param_group["params"]
            for index, param in enumerate(params):
                param_num = param.data.nelement()
                # print(param_num, param_sum, block_size)
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

        P = []
        params_packed_index = []
        for param_num in param_nums:
            if param_num >= block_size:
                block_num = math.ceil(param_num / block_size)
                for i in range(block_num):
                    if i != block_num - 1:
                        _tmp = torch.eye(block_size, dtype=self.data_type, device=self.device,)
                        params_packed_index.append(block_size)
                    else:
                        _tmp = torch.eye(param_num - block_size * i, dtype=self.data_type, device=self.device,)
                        params_packed_index.append(param_num - block_size * i)
                    if self.p0_weight is not None:
                        P.append(self.p0_weight * _tmp)
                    else:
                        P.append(_tmp)
            else:
                _tmp = torch.eye(param_num, dtype=self.data_type, device=self.device)
                params_packed_index.append(param_num)
                if self.p0_weight is not None:
                    P.append(self.p0_weight * _tmp)
                else:
                    P.append(_tmp)

        self._state.setdefault("P", P) 
        self._state.setdefault("weights_num", len(P))
        self._state.setdefault("params_packed_index", params_packed_index)
        Q = []
        if self.q0 is not None:
            for p in P:
                Q.append(
                    # self.q0 * p.clone().detach().requires_grad_(False)
                    self.q0 * torch.eye(
                                p.shape[0],
                                dtype=self.data_type,
                                device=self.device,
                            )
                    )
            self._state.setdefault("Q", Q)
            self._state.setdefault("qt", self.q0)


    def __get_blocksize(self):
        return self.param_groups[0]["block_size"]

    def __get_nue(self):
        return self.param_groups[0]["kalman_nue"]

    def __split_weights(self, weight):
        block_size = self.__get_blocksize()
        param_num = weight.nelement()
        res = []
        if param_num < block_size:
            res.append(weight)
        else:
            block_num = math.ceil(param_num / block_size)
            for i in range(block_num):
                if i != block_num - 1:
                    res.append(weight[i * block_size : (i + 1) * block_size])
                else:
                    res.append(weight[i * block_size :])
        return res

    '''
    description: 
     if q0 is not None, update params with noise, then the iters and cur_iter should not be None
    param {*} self
    param {*} H
    param {*} error
    param {*} weights
    param {*} iters: the iteration nums in epoch
    param {*} cur_iter: the current iteration, accumulated in the global epoch
    return {*}
    author: wuxingxing
    '''
    def __update(self, H, error, weights, iters=None, cur_iter=None):
        P = self._state.get("P")
        kalman_lambda = self._state.get("kalman_lambda")
        weights_num = self._state.get("weights_num")
        params_packed_index = self._state.get("params_packed_index")
        update_qt = False
        if self.q0 is not None:
            Q = self._state.get("Q")
            qt = self._state.get("qt")
            qt_next = max(self.q0*math.exp(-cur_iter/(iters*0.4343)), self.qmin)
            if (qt_next - 10**-8) > self.qmin:
                update_qt = True

        block_size = self.__get_blocksize()
        kalman_nue = self.__get_nue()

        tmp = 0
        
        for i in range(weights_num):
            if self.kalman_lambda_weight is not None:
                tmp = tmp + (self.kalman_lambda_weight * kalman_lambda + torch.matmul(torch.matmul(H[i].T, P[i]), H[i]))
            else:
                tmp = tmp + (kalman_lambda + torch.matmul(torch.matmul(H[i].T, P[i]), H[i]))
        A = 1 / tmp

        for i in range(weights_num):
            # for a Si-Si system in hybrid traing:
            # for Li-Si DP model: 0-layer [0:4050] is embding net of Li-Li, Li-Si, Si-Li, the weights are zero, \
            # 0-layer [4050:4500] is emb net of Si-Si.
            # P[0] shape is [5400, 5400] and H[0] shape is [5400, 1],  the K shape is [5400, 1].
            # when update, the K[:4050] is not zero because the H[4050:5400] is not zero.
            K = torch.matmul(P[i], H[i])

            weights[i] = weights[i] + A * error * K
            if self.q0 is None:
                P[i] = (1 / kalman_lambda) * (P[i] - A * torch.matmul(K, K.T))
            else:
                P[i] = (1 / kalman_lambda) * (P[i] - A * torch.matmul(K, K.T) + Q[i])
                if update_qt:
                    Q[i] = (qt_next/qt)*Q[i] # Q_t+1 = (q_t+1/q_t) * Qt = q_t+1 * I

        kalman_lambda = kalman_nue * kalman_lambda + 1 - kalman_nue
        self._state.update({"kalman_lambda": kalman_lambda})
        if update_qt:
            self._state.update({"qt": qt_next})
        i = 0
        param_sum = 0
        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                param_num = param.nelement()
                weight_tmp = weights[i][param_sum : param_sum + param_num]
                if param_num < block_size:
                    if param.ndim > 1:
                        param.data = weight_tmp.reshape(
                            param.data.T.shape
                        ).T.contiguous()
                    else:
                        param.data = weight_tmp.reshape(param.data.shape)

                    param_sum += param_num

                    if param_sum == params_packed_index[i]:
                        i += 1
                        param_sum = 0
                else:
                    block_num = math.ceil(param_num / block_size)
                    for j in range(block_num):
                        if j == 0:
                            tmp_weight = weights[i]
                        else:
                            tmp_weight = torch.concat([tmp_weight, weights[i]], dim=0)
                        i += 1
                    param.data = tmp_weight.reshape(param.data.T.shape).T.contiguous()

    def set_grad_prefactor(self, grad_prefactor):
        self.grad_prefactor = grad_prefactor
    
    def get_loss_l2_l1(self, lambda_l2=None, lambda_l1=None):
        L2 = torch.tensor(0.0, device=self.device, dtype=self.data_type)
        L1 = torch.tensor(0.0, device=self.device, dtype=self.data_type)
        param_nums = 0
        for idx, param in enumerate(self._params):
            if lambda_l2 is not None:
                L2 += torch.sum(param**2)
            if lambda_l1 is not None:
                L1 += torch.sum(torch.abs(param))
            param_nums += param.nelement()

        if lambda_l2 is not None:
            L2 = lambda_l2*(L2/param_nums)**0.5
        if lambda_l1 is not None:
            L1 = lambda_l1* L1/param_nums
        return L2, L1

    def print_L2(self, lambda_l2=1, lambda_l1=1):
        L2 = torch.tensor(0.0, device=self.device, dtype=self.data_type).detach().requires_grad_(False)
        L1 = torch.tensor(0.0, device=self.device, dtype=self.data_type).detach().requires_grad_(False)
        param_nums = 0
        for idx, param in enumerate(self._params):
            if lambda_l2 is not None:
                L2 += torch.sum(param**2)
            if lambda_l1 is not None:
                L1 += torch.sum(torch.abs(param))
            param_nums += param.nelement()

        if lambda_l2 is not None:
            L2 = lambda_l2*(L2/param_nums)**0.5
        if lambda_l1 is not None:
            L1 = lambda_l1* L1/param_nums
        return float(L2), float(L1)

    def step(self, error, iters, cur_iter, **kwargs):
        
        params_packed_index = self._state.get("params_packed_index")

        weights = []
        H = []
        param_index = 0
        param_sum = 0
        
        #print ("\n*************printing params*************\n")
        for idx, param in enumerate(self._params):
            
            #print(param.size())
            #print(param)
            
            #if (True in torch.isnan(param.grad)):
            #    print("nan found:")
            #    print(param.grad)
            if idx == 0 and kwargs.get('c_param') is not None:
                # get c_param for chebychev
                c_param = kwargs.get('c_param')
                c_grad = kwargs.get('c_grad')
                tmp = c_param.reshape(c_param.nelement(), 1)
                tmp_grad = c_grad.reshape(c_grad.nelement(), 1)
            else:
                if param.ndim > 1:
                    tmp = param.data.T.contiguous().reshape(param.data.nelement(), 1)
                    if param.grad is None:
                        tmp_grad = torch.zeros_like(tmp)
                    else:
                        tmp_grad = (
                            (param.grad / self.grad_prefactor)
                            .T.contiguous()
                            .reshape(param.grad.nelement(), 1)
                        )
                else:
                    tmp = param.data.reshape(param.data.nelement(), 1)
                    if param.grad is None:
                        tmp_grad = torch.zeros_like(tmp)
                    else:
                        tmp_grad = (param.grad / self.grad_prefactor).reshape(
                            param.grad.nelement(), 1
                        )

            tmp = self.__split_weights(tmp)
            tmp_grad = self.__split_weights(tmp_grad)

            for split_grad, split_weight in zip(tmp_grad, tmp):
                nelement = split_grad.nelement()

                if param_sum == 0:
                    res_grad = split_grad
                    res = split_weight
                else:
                    res_grad = torch.concat((res_grad, split_grad), dim=0)
                    res = torch.concat((res, split_weight), dim=0)

                param_sum += nelement
                
                if param_sum == params_packed_index[param_index]:
                    H.append(res_grad)
                    weights.append(res)
                    param_sum = 0
                    param_index += 1
        self.__update(H, error, weights, iters, cur_iter)
    
    """
    @Description :
    set kalman matrix p, kalman_lambda, kalman_nue when reload model.
    @Returns     :
    @Author       :wuxingxing
    """
    def set_kalman_P(self, P, kalman_lambda = 0.999999999999872):
        self._state.update({"kalman_lambda": kalman_lambda})
        for i in range(len(P)):
            self._state["P"][i] = P[i].cpu().to(self._params[0].device)

    """
    @Description :
    the kpu of energy_total and atom force: kpu = H * P * H_t, the result is a scalar
    @Returns     :
    @Author       :wuxingxing
    """
    def cal_kpu(self):
        params_packed_index = self._state.get("params_packed_index")

        weights = []    # this value doesn't need
        H = []
        param_index = 0
        param_sum = 0
        for param in self._params:
            if param.ndim > 1:
                tmp = param.data.T.contiguous().reshape(param.data.nelement(), 1)
                if param.grad is None:
                    tmp_grad = torch.zeros_like(tmp)
                else:
                    tmp_grad = (
                        (param.grad / self.grad_prefactor)
                        .T.contiguous()
                        .reshape(param.grad.nelement(), 1)
                    )
            else:
                tmp = param.data.reshape(param.data.nelement(), 1)
                if param.grad is None:
                    tmp_grad = torch.zeros_like(tmp)
                else:
                    tmp_grad = (param.grad / self.grad_prefactor).reshape(
                        param.grad.nelement(), 1
                    )

            tmp = self.__split_weights(tmp)
            tmp_grad = self.__split_weights(tmp_grad)

            for split_grad, split_weight in zip(tmp_grad, tmp):
                nelement = split_grad.nelement()

                if param_sum == 0:
                    res_grad = split_grad
                    res = split_weight
                else:
                    res_grad = torch.concat((res_grad, split_grad), dim=0)
                    res = torch.concat((res, split_weight), dim=0)

                param_sum += nelement

                if param_sum == params_packed_index[param_index]:
                    H.append(res_grad)
                    weights.append(res)
                    param_sum = 0
                    param_index += 1
        
        H_P_Ht_list = []
        for i in range(len(H)):
            H_P_Ht = torch.matmul(torch.matmul(H[i].T, self._state["P"][i]), H[i])
            H_P_Ht_list.append(H_P_Ht)
        return sum(H_P_Ht_list)

    """
    def step(self, error):

        params_packed_index = self._state.get("params_packed_index")

        weights = []
        H = []
        param_index = 0
        param_sum = 0
        
        for param in self._params:
            if (True in torch.isnan(param)):
                print("break in param")
                break
            if param.ndim > 1:
                tmp = param.data.T.contiguous().reshape(param.data.nelement(), 1)
                if (True in torch.isnan(tmp)):
                    print("break in (if) tmp")
                    break
                if param.grad is None:
                    tmp_grad = torch.zeros_like(tmp)
                else:
                    tmp_grad = (
                        (param.grad / self.grad_prefactor)
                        .T.contiguous()
                        .reshape(param.grad.nelement(), 1)
                    )
                    
                    print(param.grad)
                    if (True in torch.isnan(param.grad)):
                        print(param.grad)
                        print("break in (if) param.grad")
                        break
                    if (True in torch.isnan(tmp_grad)):
                        print("break in (if) tmp_grad")
                        break
            else:
                tmp = param.data.reshape(param.data.nelement(), 1)
                if (True in torch.isnan(tmp)):
                    print("break in tmp")
                    break
                if param.grad is None:
                    tmp_grad = torch.zeros_like(tmp)
                else:
                    tmp_grad = (param.grad / self.grad_prefactor).reshape(
                        param.grad.nelement(), 1
                    )
                    if (True in torch.isnan(param.grad)):
                        print("break in param.grad")
                        break
                    if (True in torch.isnan(tmp_grad)):
                        print("break in tmp_grad")
                        break

            tmp = self.__split_weights(tmp)
            tmp_grad = self.__split_weights(tmp_grad)
            if (True in torch.isnan(tmp[0])):
                print("break in tmp after split")
                break
            if (True in torch.isnan(tmp_grad[0])):
                print("break in tmp_grad after split")
                break

            for split_grad, split_weight in zip(tmp_grad, tmp):
                nelement = split_grad.nelement()
                # print("split_weight", split_weight[0])
                if param_sum == 0:
                    res_grad = split_grad
                    res = split_weight
                    if (True in torch.isnan(res_grad)):
                        print("break in (if) res_grad")
                        break
                    if (True in torch.isnan(res)):
                        print("break in (if) res")
                        break
                else:
                    res_grad = torch.concat((res_grad, split_grad), dim=0)
                    res = torch.concat((res, split_weight), dim=0)
                    if (True in torch.isnan(res_grad)):
                        print("break in res_grad")
                        break
                    if (True in torch.isnan(res)):
                        print("break in res")
                        break
                # print('res_grad', res_grad[0])
                param_sum += nelement

                if param_sum == params_packed_index[param_index]:
                    H.append(res_grad)
                    weights.append(res)
                    param_sum = 0
                    param_index += 1

        self.__update(H, error, weights)
    """
