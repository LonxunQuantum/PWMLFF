# `dump_observer`: observe

Test case for `observe` mode of `dump_observer`. The primary potential is the `C_2022_NEP3.txt` potential, and the secondary is a modified version of the same potential with the two first cutoffs changed. 
The resulitng `observer*.xyz` files shall be compared to the `reference_observer.xyz` file, which has been generated with `dump_exyz` using the `C_2022_NEP3` potential. 

Definition of pass: For the test of pass, the `observer0.xyz` shall match exactly to `reference_observer.xyz`, and `observer1.xyz` shall not match.

## Changelog 31-01-2024
Updated the `reference_observer.xyz` file by running `dump_exyz` with the `C_2022_NEP3.txt` potential again. Something has changed in GPUMD which broke this consistency check. I unfortunately don't know when it happened, so I don't know if it was beneficial or not.
