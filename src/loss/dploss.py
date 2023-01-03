import numpy as np


def dp_loss(
    start_lr,
    real_lr,
    has_fi,
    lossFi,
    has_etot,
    loss_Etot,
    has_egroup,
    loss_Egroup,
    has_ei,
    loss_Ei,
    natoms_sum,
):
    start_pref_egroup, limit_pref_egroup = 0.02, 1.0
    start_pref_F, limit_pref_F = 1000, 1.0
    start_pref_etot, limit_pref_etot = 0.02, 1.0
    start_pref_ei, limit_pref_ei = 0.02, 1.0
    pref_fi = has_fi * (
        limit_pref_F + (start_pref_F - limit_pref_F) * real_lr / start_lr
    )
    pref_etot = has_etot * (
        limit_pref_etot + (start_pref_etot - limit_pref_etot) * real_lr / start_lr
    )
    pref_egroup = has_egroup * (
        limit_pref_egroup + (start_pref_egroup - limit_pref_egroup) * real_lr / start_lr
    )
    pref_ei = has_ei * (
        limit_pref_ei + (start_pref_ei - limit_pref_ei) * real_lr / start_lr
    )
    l2_loss = 0
    if has_fi:
        l2_loss += pref_fi * lossFi
    if has_etot:
        l2_loss += 1.0 / natoms_sum * pref_etot * loss_Etot
    if has_egroup:
        l2_loss += pref_egroup * loss_Egroup
    if has_ei:
        l2_loss += pref_ei * loss_Ei
    return l2_loss, pref_fi, pref_etot


def adjust_lr(iter, start_lr, stop_lr=3.51e-8):
    stop_step = 1000000
    decay_step = 5000
    decay_rate = np.exp(
        np.log(stop_lr / start_lr) / (stop_step / decay_step)
    )  # 0.9500064099092085
    real_lr = start_lr * np.power(decay_rate, (iter // decay_step))
    return real_lr
