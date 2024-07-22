import torch
from feature.chebyshev.basic import Radial, SmoothFunc

class Descriptor:
    def __init__(self, batch_size, beta, m1, m2, rcut_max, rcut_smooth,
                 natoms, ntypes, max_neighbors, type_map,
                 num_neigh, neighbors_list, dr_neigh, c, device, dtype):
        self.batch_size = batch_size
        self.beta = beta
        self.m1 = m1
        self.m2 = m2
        self.rcut_max = rcut_max
        self.rcut_smooth = rcut_smooth
        self.natoms = natoms
        self.ntypes = ntypes
        self.max_neighbors = max_neighbors
        self.type_map = type_map                    # shape: torch.Size([natoms])
        self.num_neigh = num_neigh                  # shape: torch.Size([bt, natoms, ntypes])
        self.neighbors_list = neighbors_list        # shape: torch.Size([bt, natoms, ntypes, max_neighbors])
        self.dr_neigh = dr_neigh                    # shape: torch.Size([bt, natoms, ntypes, max_neighbors, 4])
        self.c = c                                  # shape: torch.Size([ntypes, ntypes, m1, beta])
        self.device = device
        self.dtype = dtype

        self.radial = Radial(m1, beta, ntypes, rcut_max, rcut_smooth, c)
        self.smooth = SmoothFunc(rcut_max, rcut_smooth)

        self.nfeat = m1 * m2
        self.feats = torch.zeros((batch_size, natoms, self.nfeat), dtype=self.dtype, device=self.device)
        # self.dfeats = torch.zeros((batch_size, natoms, self.nfeat * max_neighbors, 3), dtype=self.dtype, device=self.device)
        self.dfeats = torch.zeros((batch_size, natoms, self.nfeat, max_neighbors, 3), dtype=self.dtype, device=self.device)
        self.build(batch_size, max_neighbors, ntypes, natoms)

    def build(self, batch_size, max_neighbors, ntypes, natoms):
        ind_neigh_alltypes = torch.zeros((batch_size, natoms, ntypes, max_neighbors), dtype=torch.int32, device=self.device)

        mask = torch.arange(max_neighbors, device=self.device).view(1, 1, 1, max_neighbors) < self.num_neigh.unsqueeze(-1)
        neighbors_list = torch.where(mask, self.neighbors_list, torch.tensor(999, device=self.device, dtype=torch.int32)).view(batch_size, natoms, -1)
        sorted_neighbors, _ = torch.sort(neighbors_list, dim=-1, descending=False)
        neighbor_list_alltypes = sorted_neighbors[:, :, :max_neighbors]
        neighbor_list_alltypes = torch.where(neighbor_list_alltypes == 999, torch.tensor(-1, device=self.device, dtype=torch.int32), neighbor_list_alltypes)
        self.neighbor_list_alltypes = neighbor_list_alltypes
        
        # Calculate the starting index of each type
        type_offsets = torch.cumsum(self.num_neigh, dim=-1) - self.num_neigh
        type_offsets_expanded = type_offsets.unsqueeze(-1).expand(batch_size, natoms, ntypes, max_neighbors)
        neighbor_expanded = torch.arange(max_neighbors, device=self.device).view(1, 1, 1, max_neighbors).expand(batch_size, natoms, ntypes, max_neighbors)
        if (neighbor_expanded >= max_neighbors).any():
            raise ValueError('Error: the maximum number of neighbors is too small.')
        ind_neigh_alltypes[mask] = (type_offsets_expanded[mask] + neighbor_expanded[mask]).int()
        num_neigh_alltypes = self.num_neigh.sum(dim=2)


        rij_test = self.dr_neigh[:, :, :, :, 0][mask]
        delx_test = self.dr_neigh[:, :, :, :, 1][mask]
        dely_test = self.dr_neigh[:, :, :, :, 2][mask]
        delz_test = self.dr_neigh[:, :, :, :, 3][mask]
        # jj_test = ind_neigh_alltypes[:, :, :][mask]

        '''
        jtype = []
        for bt in range(batch_size):
            for i in range(natoms):
                indices = self.num_neigh[bt, i]
                temp = []
                for idx, count in enumerate(indices):
                    if count > 0:
                        temp.extend([idx] * count)
                jtype += temp
        '''

        jtype_test = torch.cat([torch.cat([torch.arange(self.num_neigh.size(2), device=self.device).repeat_interleave(self.num_neigh[bt, i]) 
                                        for i in range(self.num_neigh.size(1))]) 
                                        for bt in range(self.num_neigh.size(0))])
        
        rads_test, drads_test, fc_test, dfc_test = self.radial.build3(natoms, rij_test, self.type_map, jtype_test)
        s_test = fc_test / rij_test
        nneigh_test = num_neigh_alltypes.flatten().cumsum(dim=0)
        ff = (dfc_test / rij_test - fc_test / (rij_test * rij_test)).unsqueeze(-1) * rads_test + s_test.unsqueeze(-1) * drads_test
        """
        for bt in range(batch_size):
            for i in range(natoms):
                bt_i = bt * natoms + i
                nneigh = num_neigh_alltypes.flatten()[bt_i]
                start_idx = 0 if bt_i == 0 else nneigh_test[bt_i - 1].item()
                end_idx = nneigh_test[bt_i].item()
                rads_idx = rads_test[i, start_idx:end_idx]
                s_idx = s_test[start_idx:end_idx]
                delx_idx = delx_test[start_idx:end_idx]
                dely_idx = dely_test[start_idx:end_idx]
                delz_idx = delz_test[start_idx:end_idx]
                rij_idx = rij_test[start_idx:end_idx]
                jj_idx = jj_test[start_idx:end_idx]
                ff_idx = ff[i, start_idx:end_idx]

                T_test = torch.zeros((self.m1, 4), dtype=self.dtype, device=self.device)
                dT_test = torch.zeros((3, nneigh, self.m1, 4), dtype=self.dtype, device=self.device)
                T_test[:, 0] = torch.sum((rads_idx * s_idx.unsqueeze(-1)), dim=0)
                T_test[:, 1] = torch.sum((rads_idx * (s_idx * delx_idx / rij_idx).unsqueeze(-1)), dim=0)
                T_test[:, 2] = torch.sum((rads_idx * (s_idx * dely_idx / rij_idx).unsqueeze(-1)), dim=0)
                T_test[:, 3] = torch.sum((rads_idx * (s_idx * delz_idx / rij_idx).unsqueeze(-1)), dim=0)

                dT_test[0, jj_idx, :, 0] = ff_idx * (delx_idx / rij_idx).unsqueeze(-1)
                dT_test[1, jj_idx, :, 0] = ff_idx * (dely_idx / rij_idx).unsqueeze(-1)
                dT_test[2, jj_idx, :, 0] = ff_idx * (delz_idx / rij_idx).unsqueeze(-1)
                
                comp1 = rads_idx * (s_idx / rij_idx).unsqueeze(-1)
                comp2 = ff_idx - comp1
                dT_test[0, jj_idx, :, 1] = (comp2) * (delx_idx * delx_idx / (rij_idx * rij_idx)).unsqueeze(-1)
                dT_test[1, jj_idx, :, 1] = (comp2) * (delx_idx * dely_idx / (rij_idx * rij_idx)).unsqueeze(-1)
                dT_test[2, jj_idx, :, 1] = (comp2) * (delx_idx * delz_idx / (rij_idx * rij_idx)).unsqueeze(-1)
                dT_test[0, jj_idx, :, 1] += comp1

                dT_test[0, jj_idx, :, 2] = (comp2) * (dely_idx * delx_idx / (rij_idx * rij_idx)).unsqueeze(-1)
                dT_test[1, jj_idx, :, 2] = (comp2) * (dely_idx * dely_idx / (rij_idx * rij_idx)).unsqueeze(-1)
                dT_test[2, jj_idx, :, 2] = (comp2) * (dely_idx * delz_idx / (rij_idx * rij_idx)).unsqueeze(-1)
                dT_test[1, jj_idx, :, 2] += comp1

                dT_test[0, jj_idx, :, 3] = (comp2) * (delz_idx * delx_idx / (rij_idx * rij_idx)).unsqueeze(-1)
                dT_test[1, jj_idx, :, 3] = (comp2) * (delz_idx * dely_idx / (rij_idx * rij_idx)).unsqueeze(-1)
                dT_test[2, jj_idx, :, 3] = (comp2) * (delz_idx * delz_idx / (rij_idx * rij_idx)).unsqueeze(-1)
                dT_test[2, jj_idx, :, 3] += comp1

                # version 1
                # for ii1 in range(self.m1):
                #     for ii2 in range(self.m2):
                #         dsum = ((dT_test[:, :, ii1] * T_test[ii2]).sum(-1) + (T_test[ii1] * dT_test[:, :, ii2]).sum(-1)).permute(1, 0)
                #         ii = ii1 * self.m2 + ii2
                #         self.feats[bt_i, ii] += (T_test[ii1] * T_test[ii2]).sum(-1)
                #         mask_d = dsum.abs().sum(dim=1) > 1e-7
                #         indices = torch.nonzero(mask_d, as_tuple=False).squeeze()
                #         self.dfeats[bt_i, ii, indices] = dsum[mask_d]

                ii1_range = torch.arange(self.m1, device=self.device).view(-1, 1)
                ii2_range = torch.arange(self.m2, device=self.device).view(1, -1)
                ii_range = ii1_range * self.m2 + ii2_range

                dsum = ((dT_test[:, :, ii1_range] * T_test[ii2_range]).sum(-1) + (T_test[ii1_range] * dT_test[:, :, ii2_range]).sum(-1)).permute(2, 3, 1, 0)
                dsum = dsum.reshape(self.nfeat, -1, 3)
                self.feats[bt_i, ii_range] += (T_test[ii1_range] * T_test[ii2_range]).sum(-1)   #feat: batch_size * natoms, self.nfeat
                mask_d = dsum.abs().sum(dim=2) > 1e-7
                dsum_mask = dsum[mask_d]
                indices = torch.nonzero(mask_d, as_tuple=False).squeeze()
                idx_offsets = indices[:, 1] + indices[:, 0] * max_neighbors
                self.dfeats[bt_i].scatter_add_(0, idx_offsets.unsqueeze(1).repeat(1,3), dsum_mask)  #dfeat: batch_size * natoms, self.nfeat, max_neighbors, 3

                #version 2
                # _, counts = torch.unique(indices[:, 0], return_counts=True)
                # groups = torch.split(indices[:, 1], counts.tolist())
                # for ii in ii_range.flatten():
                #     start_idx = 0 if ii == 0 else counts[:ii].sum().item()
                #     end_idx = counts[:ii + 1].sum().item()
                #     self.dfeats[bt_i, ii, groups[ii]] = dsum_mask[start_idx:end_idx]  #dfeat: batch_size * natoms, self.nfeat, max_neighbors, 3

        self.feats = self.feats.view(batch_size, natoms, self.nfeat)
        self.dfeats = self.dfeats.view(batch_size, natoms, self.nfeat, max_neighbors, 3)
        """
        # 以下是去掉部分循环的代码
        start_indices = torch.cat([torch.tensor([0], device=self.device), nneigh_test[:-1]])
        end_indices = nneigh_test
        T_test = torch.zeros((natoms, len(jtype_test), self.m1, 4), dtype=self.dtype, device=self.device)
        dT_test = torch.zeros((natoms, 3, len(jtype_test), self.m1, 4), dtype=self.dtype, device=self.device)
        
        range_masks = torch.zeros((natoms, nneigh_test[-1]), dtype=torch.bool, device=self.device)
        for bt in range(batch_size):
            for i in range(natoms):
                bt_i = bt * natoms + i
                range_masks[i, start_indices[bt_i]:end_indices[bt_i]] = True

        rads_s = rads_test * s_test.unsqueeze(-1)
        rads_s_delx_rij = rads_test * (s_test * delx_test / rij_test).unsqueeze(-1)
        rads_s_dely_rij = rads_test * (s_test * dely_test / rij_test).unsqueeze(-1)
        rads_s_delz_rij = rads_test * (s_test * delz_test / rij_test).unsqueeze(-1)
        T_test[:, :, :, 0] = rads_s * range_masks.unsqueeze(2)
        T_test[:, :, :, 1] = rads_s_delx_rij * range_masks.unsqueeze(2)
        T_test[:, :, :, 2] = rads_s_dely_rij * range_masks.unsqueeze(2)
        T_test[:, :, :, 3] = rads_s_delz_rij * range_masks.unsqueeze(2)


        dT_test[:, 0, :, :, 0] = (ff * (delx_test / rij_test).unsqueeze(-1) * range_masks.unsqueeze(2))
        dT_test[:, 1, :, :, 0] = (ff * (dely_test / rij_test).unsqueeze(-1) * range_masks.unsqueeze(2))
        dT_test[:, 2, :, :, 0] = (ff * (delz_test / rij_test).unsqueeze(-1) * range_masks.unsqueeze(2))

        comp1 = rads_test * (s_test / rij_test).unsqueeze(-1)
        comp2 = ff - comp1
        dT_test[:, 0, :, :, 1] = (comp2 * (delx_test * delx_test / (rij_test * rij_test)).unsqueeze(-1) * range_masks.unsqueeze(2))
        dT_test[:, 1, :, :, 1] = (comp2 * (delx_test * dely_test / (rij_test * rij_test)).unsqueeze(-1) * range_masks.unsqueeze(2))
        dT_test[:, 2, :, :, 1] = (comp2 * (delx_test * delz_test / (rij_test * rij_test)).unsqueeze(-1) * range_masks.unsqueeze(2))
        dT_test[:, 0, :, :, 1] += comp1

        dT_test[:, 0, :, :, 2] = (comp2 * (dely_test * delx_test / (rij_test * rij_test)).unsqueeze(-1) * range_masks.unsqueeze(2))
        dT_test[:, 1, :, :, 2] = (comp2 * (dely_test * dely_test / (rij_test * rij_test)).unsqueeze(-1) * range_masks.unsqueeze(2))
        dT_test[:, 2, :, :, 2] = (comp2 * (dely_test * delz_test / (rij_test * rij_test)).unsqueeze(-1) * range_masks.unsqueeze(2))
        dT_test[:, 1, :, :, 2] += comp1

        dT_test[:, 0, :, :, 3] = (comp2 * (delz_test * delx_test / (rij_test * rij_test)).unsqueeze(-1) * range_masks.unsqueeze(2))
        dT_test[:, 1, :, :, 3] = (comp2 * (delz_test * dely_test / (rij_test * rij_test)).unsqueeze(-1) * range_masks.unsqueeze(2))
        dT_test[:, 2, :, :, 3] = (comp2 * (delz_test * delz_test / (rij_test * rij_test)).unsqueeze(-1) * range_masks.unsqueeze(2))
        dT_test[:, 2, :, :, 3] += comp1

        # feats = torch.zeros((batch_size, natoms, self.nfeat), dtype=self.dtype, device=self.device)
        # dfeats = torch.zeros((batch_size, natoms, self.nfeat, max_neighbors, 3), dtype=self.dtype, device=self.device)
        ii1_range = torch.arange(self.m1, device=self.device).view(-1, 1)
        ii2_range = torch.arange(self.m2, device=self.device).view(1, -1)
        """
        for bt in range(batch_size):
            for i in range(natoms):
                bt_i = bt * natoms + i
                start_idx = 0 if bt_i == 0 else nneigh_test[bt_i - 1].item()
                end_idx = nneigh_test[bt_i].item()
                T_test_ = T_test[i, start_idx:end_idx].sum(0)
                dT_test_ = dT_test[i, :, start_idx:end_idx]
                self.feats[bt, i] += (T_test_[ii1_range] * T_test_[ii2_range]).sum(-1).reshape(self.nfeat)
                dsum = ((dT_test_[:, :, ii1_range] * T_test_[ii2_range]).sum(-1) + (T_test_[ii1_range] * dT_test_[:, :, ii2_range]).sum(-1)).permute(2, 3, 1, 0)
                dsum = dsum.reshape(self.nfeat, -1, 3)
                mask_d = dsum.abs().sum(dim=2) > 1e-7
                dsum_mask = dsum[mask_d]
                indices = torch.nonzero(mask_d, as_tuple=False).squeeze()
                idx_offsets = indices[:, 1] + indices[:, 0] * max_neighbors
                self.dfeats[bt, i].scatter_add_(0, idx_offsets.unsqueeze(1).repeat(1,3), dsum_mask)     #dfeat: batch_size, natoms, self.nfeat * max_neighbors, 3
                
        self.dfeats = self.dfeats.view(batch_size, natoms, self.nfeat, max_neighbors, 3)   
        """

        _sizes = num_neigh_alltypes.sum(1)
        cum_sizes = num_neigh_alltypes.cumsum(dim=1)
        bt_start = 0
        for i, size in enumerate(_sizes):
            bt_end = bt_start + size
            num_neigh_batch = num_neigh_alltypes[i]
            cum_sizes_bt = cum_sizes[i]
            range_masks_ = range_masks[:, bt_start:bt_end]
            T_test_ = (T_test[:, bt_start:bt_end] * range_masks_.unsqueeze(2).unsqueeze(3)).sum(1)
            dT_test_ = dT_test[:, :, bt_start:bt_end] * range_masks_.unsqueeze(1).unsqueeze(3).unsqueeze(4)
            self.feats[i] = (T_test_[:, ii1_range] * T_test_[:, ii2_range]).sum(-1).reshape(natoms, self.nfeat)
            dsum = ((dT_test_[:, :, :, ii1_range] * T_test_[:, ii2_range].unsqueeze(1).unsqueeze(2)).sum(-1) + (T_test_[:, ii1_range].unsqueeze(1).unsqueeze(2) * dT_test_[:, :, :, ii2_range]).sum(-1)).permute(0, 2, 3, 4, 1)
            dsum_mask = dsum[dsum.sum(-1) != 0].reshape(-1, self.nfeat, 3)
            result = torch.zeros(natoms, max_neighbors, self.nfeat, 3, dtype=self.dtype, device=self.device)
            cum_sizes_bt_expanded = cum_sizes_bt.unsqueeze(1).expand(natoms, max_neighbors)
            index_range = torch.arange(max_neighbors, device=self.device).unsqueeze(0).expand(natoms, max_neighbors)
            group_sizes_expanded = num_neigh_batch.unsqueeze(1).expand(natoms, max_neighbors)
            valid_mask = index_range < group_sizes_expanded
            source_indices = (cum_sizes_bt_expanded - group_sizes_expanded + index_range)[valid_mask]
            result[valid_mask] = dsum_mask[source_indices]
            result = result.permute(0, 2, 1, 3)
            abs_sum = result.abs().sum(dim=3, keepdim=True) > 1e-7
            self.dfeats[i] = result * abs_sum
            bt_start = bt_end

        '''
        for bt in range(batch_size):
            for i in range(natoms):
                nneigh = num_neigh_alltypes[bt, i]
                itype = self.type_map[i]

                T = torch.zeros((self.m1 * ntypes, 4), dtype=self.dtype, device=self.device)
                dT = torch.zeros((3, nneigh, self.m1 * ntypes, 4), dtype=self.dtype, device=self.device)
                
                for jtype in range(ntypes):
                    for j in range(self.num_neigh[bt, i, jtype]):
                        jj = ind_neigh_alltypes[bt, i, jtype, j]
                        rij = self.dr_neigh[bt, i, jtype, j, 0]
                        delx = self.dr_neigh[bt, i, jtype, j, 1]
                        dely = self.dr_neigh[bt, i, jtype, j, 2]
                        delz = self.dr_neigh[bt, i, jtype, j, 3]
                        rads, drads = self.radial.build(rij, itype, jtype)
                        fc = self.smooth.get_smooth(rij)
                        dfc = self.smooth.get_dsmooth(rij)
                        s = fc / rij
                        for m in range(self.m1):
                            ii = m + itype * self.m1
                            T, dT = self.build_components(m, ii, jj, delx, dely, delz, 
                                                                        itype, jtype, rads, drads, 
                                                                        fc, dfc, s, rij, T, dT)
                
                for ii1 in range(self.m1):
                    index_m1 = itype * self.m1 + ii1
                    for ii2 in range(self.m2):
                        index_m2 = itype * self.m1 + ii2

                        dsum = ((dT[:, :, index_m1] * T[index_m2]).sum(-1) + (T[index_m1] * dT[:, :, index_m2]).sum(-1)).permute(1, 0)

                        ii = ii1 * self.m2 + ii2
                        self.feats[bt, i, ii] += (T[index_m1] * T[index_m2]).sum(-1)

                        mask_d = dsum.abs().sum(dim=1) > 1e-7
                        indices = torch.nonzero(mask_d, as_tuple=False).squeeze()
                        self.dfeats[bt, i, ii, indices] = dsum[mask_d]
        '''
    
    def build_components(self, m, ii, jj, delx, dely, delz, itype, jtype, rads, drads, fc, dfc, s, rij, T, dT):
        T[ii, 0] += rads[itype, jtype, m] * s
        T[ii, 1] += rads[itype, jtype, m] * s * delx / rij
        T[ii, 2] += rads[itype, jtype, m] * s * dely / rij
        T[ii, 3] += rads[itype, jtype, m] * s * delz / rij

        ff = (dfc / rij - fc / (rij * rij)) * rads[itype, jtype, m] + s * drads[itype, jtype, m]

        dT[0, jj, ii, 0] += ff * delx / rij
        dT[1, jj, ii, 0] += ff * dely / rij
        dT[2, jj, ii, 0] += ff * delz / rij

        dT[0, jj, ii, 1] += (ff - s * rads[itype, jtype, m] / rij) * delx * delx / (rij * rij)
        dT[1, jj, ii, 1] += (ff - s * rads[itype, jtype, m] / rij) * delx * dely / (rij * rij)
        dT[2, jj, ii, 1] += (ff - s * rads[itype, jtype, m] / rij) * delx * delz / (rij * rij)
        dT[0, jj, ii, 1] += rads[itype, jtype, m] * s / rij

        dT[0, jj, ii, 2] += (ff - s * rads[itype, jtype, m] / rij) * dely * delx / (rij * rij)
        dT[1, jj, ii, 2] += (ff - s * rads[itype, jtype, m] / rij) * dely * dely / (rij * rij)
        dT[2, jj, ii, 2] += (ff - s * rads[itype, jtype, m] / rij) * dely * delz / (rij * rij)
        dT[1, jj, ii, 2] += rads[itype, jtype, m] * s / rij

        dT[0, jj, ii, 3] += (ff - s * rads[itype, jtype, m] / rij) * delz * delx / (rij * rij)
        dT[1, jj, ii, 3] += (ff - s * rads[itype, jtype, m] / rij) * delz * dely / (rij * rij)
        dT[2, jj, ii, 3] += (ff - s * rads[itype, jtype, m] / rij) * delz * delz / (rij * rij)
        dT[2, jj, ii, 3] += rads[itype, jtype, m] * s / rij
        return T, dT