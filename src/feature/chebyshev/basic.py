import torch

class CalcKsi:
    def __init__(self, rcut_max, rcut_smooth) -> None:
        self.rcut_max = rcut_max
        self.rcut_smooth = rcut_smooth

    def get_si(self, rij):
        return (2.0 * rij - (self.rcut_max + self.rcut_smooth)) / (self.rcut_max - self.rcut_smooth)
    
    def get_dsi(self):
        return 2.0 / (self.rcut_max - self.rcut_smooth)
    

class Chebyshev1st:
    def __init__(self, beta, rcut_max, rcut_smooth) -> None:
        self.beta = beta
        self.rcut_max = rcut_max
        self.rcut_smooth = rcut_smooth
        self.ksi = CalcKsi(rcut_max, rcut_smooth)

    def build(self, rij):
        vals = torch.zeros(self.beta, dtype=rij.dtype, device=rij.device)
        ders = torch.zeros_like(vals)
        ders2r = torch.zeros_like(vals)
        si = self.ksi.get_si(rij)
        if ((rij >= self.rcut_smooth) and (rij <= self.rcut_max)):
            vals[0] = 1.0
            vals[1] = si
            ders[0] = 0.0
            ders[1] = 1.0
            ders2r[0] = 0.0
            ders2r[1] = 1.0 * self.ksi.get_dsi()
            for i in range(2, self.beta):
                vals[i] = 2.0 * si * vals[i - 1] - vals[i - 2]
                ders[i] = 2.0 * si * ders[i - 1] + 2.0 * vals[i - 1] - ders[i - 2]
                ders2r[i] = ders[i] * self.ksi.get_dsi() 
        return vals, ders2r
    
    def build2(self, rij):
        vals = torch.zeros(rij.shape[0], self.beta, dtype=rij.dtype, device=rij.device)
        ders = torch.zeros_like(vals)
        ders2r = torch.zeros_like(vals)
        si = self.ksi.get_si(rij)
        mask = (rij >= self.rcut_smooth) & (rij <= self.rcut_max)
        vals[mask, 0] = 1.0
        ders[mask, 0] = 0.0
        vals[mask, 1] = si[mask]
        ders[mask, 1] = 1.0
        ders2r[mask, 0] = 0.0
        ders2r[mask, 1] = 1.0 * self.ksi.get_dsi()
        for i in range(2, self.beta):
            vals[mask, i] = 2.0 * si[mask] * vals[mask, i - 1] - vals[mask, i - 2]
            ders[mask, i] = 2.0 * si[mask] * ders[mask, i - 1] + 2.0 * vals[mask, i - 1] - ders[mask, i - 2]
            ders2r[mask, i] = ders[mask, i] * self.ksi.get_dsi()
        return vals, ders2r


class SmoothFunc:
    def __init__(self, rcut_max, rcut_smooth) -> None:
        self.rcut_max = rcut_max
        self.rcut_smooth = rcut_smooth

    def get_smooth(self, rij):
        u = (rij - self.rcut_smooth) / (self.rcut_max - self.rcut_smooth)
        if (rij <= self.rcut_smooth):
            fc = 1.0
        elif (rij >= self.rcut_smooth) and (rij < self.rcut_max):
            fc = torch.pow(u, 3) * (-6.0 * torch.pow(u, 2) + 15.0 * u - 10.0) + 1.0
        else:
            fc = 0.0
        return fc
    
    def get_dsmooth(self, rij):
        u = (rij - self.rcut_smooth) / (self.rcut_max - self.rcut_smooth)
        if (rij <= self.rcut_smooth):
            dfc = 0.0
        elif (rij >= self.rcut_smooth) and (rij < self.rcut_max):
            dfc = 1.0 / (self.rcut_max - self.rcut_smooth) * (-30.0 * torch.pow(u, 4) + 60.0 * torch.pow(u, 3) - 30.0 * torch.pow(u, 2))
        else:
            dfc = 0.0
        return dfc
    
    def get_smooth2(self, rij):
        u = (rij - self.rcut_smooth) / (self.rcut_max - self.rcut_smooth)
        mask_lt = rij <= self.rcut_smooth
        mask_eq = (rij >= self.rcut_smooth) & (rij < self.rcut_max)
        mask_gt = rij >= self.rcut_max
        fc = torch.zeros_like(rij)
        dfc = torch.zeros_like(rij)
        fc[mask_lt] = 1.0
        fc[mask_eq] = torch.pow(u, 3) * (-6.0 * torch.pow(u, 2) + 15.0 * u - 10.0) + 1.0
        fc[mask_gt] = 0.0
        dfc[mask_lt] = 0.0
        dfc[mask_eq] = 1.0 / (self.rcut_max - self.rcut_smooth) * (-30.0 * torch.pow(u, 4) + 60.0 * torch.pow(u, 3) - 30.0 * torch.pow(u, 2))
        dfc[mask_gt] = 0.0
        return fc, dfc
    
    def get_smooth3(self, rij):
        mask_lt = rij <= self.rcut_smooth
        mask_eq = (rij >= self.rcut_smooth) & (rij < self.rcut_max)
        mask_gt = rij >= self.rcut_max
        fc = torch.zeros_like(rij)
        dfc = torch.zeros_like(rij)
        fc[mask_lt] = 1.0
        fc[mask_eq] = 1/2 * (torch.cos(3.141592653589793 * (rij - self.rcut_smooth) / (self.rcut_max - self.rcut_smooth)) + 1)
        fc[mask_gt] = 0.0
        dfc[mask_lt] = 0.0
        dfc[mask_eq] = -3.141592653589793 / (2 * (self.rcut_max - self.rcut_smooth)) * torch.sin(3.141592653589793 * (rij - self.rcut_smooth) / (self.rcut_max - self.rcut_smooth))
        dfc[mask_gt] = 0.0
        return fc, dfc
        

class Radial:
    def __init__(self, mu, beta, ntypes, rcut_max, rcut_smooth, c) -> None:
        self.mu = mu
        self.beta = beta
        self.ntypes = ntypes
        self.rcut_max = rcut_max
        self.rcut_smooth = rcut_smooth
        self.c = c
        self.smooth = SmoothFunc(rcut_max, rcut_smooth)
        self.chebyshev = Chebyshev1st(beta, rcut_max, rcut_smooth)

    def build(self, rij, itype, jtype):
        fc = self.smooth.get_smooth(rij)
        dfc = self.smooth.get_dsmooth(rij)
        vals, ders2r = self.chebyshev.build(rij)
        rads = torch.zeros((self.ntypes, self.ntypes, self.mu), dtype=rij.dtype, device=rij.device)
        drads = torch.zeros_like(rads)
        for m in range(self.mu):
            rads[itype, jtype, m] = 0.0
            drads[itype, jtype, m] = 0.0
            for n in range(self.beta):
                rads[itype, jtype, m] += vals[n] * fc * self.c[itype, jtype, m, n]
                drads[itype, jtype, m] += (ders2r[n] * fc + vals[n] * dfc) * self.c[itype, jtype, m, n]

        return rads, drads

    def build2(self, natoms, rij, itype, jtype):
        fc, dfc = self.smooth.get_smooth2(rij)
        vals, ders2r = self.chebyshev.build2(rij)
        c_itype = self.c[itype]
        c_jtype = c_itype[:, jtype]
        rads = torch.sum((vals.unsqueeze(0).unsqueeze(2).repeat(natoms, 1, self.mu, 1) * c_jtype), dim=-1) * fc.unsqueeze(-1)
        drads = torch.sum((ders2r.unsqueeze(0).unsqueeze(2).repeat(natoms, 1, self.mu, 1) * c_jtype), dim=-1) * fc.unsqueeze(-1) + torch.sum((vals.unsqueeze(0).unsqueeze(2).repeat(natoms,1,self.mu,1) * c_jtype), dim=-1) * dfc.unsqueeze(-1)
        return rads, drads, fc, dfc
    
    def build3(self, natoms, rij, itype, jtype):
        fc, dfc = self.smooth.get_smooth3(rij)
        vals, ders2r = self.chebyshev.build2(rij)
        c_itype = self.c[itype]
        c_jtype = c_itype[:, jtype]
        rads = 0.5 *torch.sum(((vals.unsqueeze(0).unsqueeze(2).repeat(natoms, 1, self.mu, 1) * c_jtype)*(2*((rij - self.rcut_smooth)/(self.rcut_max - self.rcut_smooth) -1)**2 -1).unsqueeze(0).unsqueeze(2).unsqueeze(3) + 1), dim=-1) * fc.unsqueeze(-1)
        drads = 0.5 *(torch.sum(((ders2r.unsqueeze(0).unsqueeze(2).repeat(natoms, 1, self.mu, 1) * c_jtype)*(2*((rij - self.rcut_smooth)/(self.rcut_max - self.rcut_smooth) -1)**2 -1).unsqueeze(0).unsqueeze(2).unsqueeze(3) + 1), dim=-1) + \
                      torch.sum(((vals.unsqueeze(0).unsqueeze(2).repeat(natoms,1,self.mu,1) * c_jtype)*(4*((rij - self.rcut_smooth)/(self.rcut_max - self.rcut_smooth) -1)*(1/(self.rcut_max - self.rcut_smooth))).unsqueeze(0).unsqueeze(2).unsqueeze(3)), dim=-1)) * fc.unsqueeze(-1) +\
                0.5 *torch.sum(((vals.unsqueeze(0).unsqueeze(2).repeat(natoms, 1, self.mu, 1) * c_jtype)*(2*((rij - self.rcut_smooth)/(self.rcut_max - self.rcut_smooth) -1)**2 -1).unsqueeze(0).unsqueeze(2).unsqueeze(3) + 1), dim=-1) * dfc.unsqueeze(-1)
        return rads, drads, fc, dfc