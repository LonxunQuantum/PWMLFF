from subprocess import run
from typing import List

import numpy as np
import pandas as pd
from ase.io import read


def _copy_files(files: List[str], test_folder: str, tmp_path: str):
    for file in files:
        with open(f'{test_folder}/{file}', 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        content = '\n'.join(lines)
        p = tmp_path / file
        p.write_text(content)


def _load_observer_files(path: str, mode: str):
    n_output_files = 1 if mode == 'average' else 2
    n_reference_files = 2 if mode == 'average' else 2
    data = []
    if n_output_files == 1:
        bundles = (read(f'{path}/observer.xyz', ':'), np.loadtxt(f'{path}/observer.out'))
    else:
        bundles = ()
        for i in range(n_output_files):
            bundles += (read(f'{path}/observer{i}.xyz', ':'), np.loadtxt(f'{path}/observer{i}.out'))
            # bundles += ([1*i,2,3], [1*i,2,3])
    for k, bundle in enumerate(zip(*bundles)):
        data_bundle = {
                'timestep': k,
        }
        # Group bundle into twos
        frame_bundle = [(bundle[i], bundle[i+1]) for i in range(0, len(bundle), 2)]
        for m, (frame, row) in enumerate(frame_bundle):
            energy = frame.get_potential_energy()
            forces = frame.get_forces()
            data_bundle[f'energy{m}_exyz'] = energy
            data_bundle[f'energy{m}_thermo'] = row[2]
            data_bundle[f'forces{m}_exyz'] = forces.flatten()
        data.append(data_bundle)
    df = pd.DataFrame.from_dict(data)
    
    ref_structures = [read(f'{path}/reference_observer{i}.xyz', ':') for i in range(n_reference_files)]
    ref_data = []
    for refs in zip(*ref_structures):
        frame_data = {}
        for i, ref in enumerate(refs):
            frame_data[f'energy_ref{i}']= ref.get_potential_energy()
            frame_data[f'forces_ref{i}']= ref.get_forces().flatten()
        ref_data.append(frame_data)

    ref_df = pd.DataFrame.from_dict(ref_data)
    return df, ref_df


def test_observe_single_species(tmp_path):
    test_folder = 'gpumd/dump_observer/carbon_observe'
    files = [
        'C_2022_NEP3.txt',
        'C_2022_NEP3_MODIFIED.txt',
        'model.xyz',
        'reference_observer0.xyz',
        'reference_observer1.xyz',
        'run.in'
    ]
    _copy_files(files, test_folder, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    df, ref_df = _load_observer_files(tmp_path, mode='observe')
    atol = 1e-8  # Anything smaller is considered ~0
    rtol = 1e-7
    assert not np.any(np.isclose(
        a=df['energy0_exyz'],
        b=df['energy1_exyz'],
        atol=atol,
        rtol=rtol
    )), 'Energies should be different'
    assert not np.any(np.isclose(
        a=df['energy0_thermo'],
        b=df['energy1_thermo'],
        atol=atol,
        rtol=rtol
    )), 'Energies should be different'
    assert np.all(np.isclose(
        a=df['energy0_exyz'],
        b=df['energy0_thermo'],
        atol=atol,
        rtol=rtol
    )), 'Energies should be consistent between exyz and thermo'
    assert np.all(np.isclose(
        a=df['energy1_exyz'],
        b=df['energy1_thermo'],
        atol=atol,
        rtol=rtol
    )), 'Energies should be consistent between exyz and thermo'

    forces0 = np.concatenate(df['forces0_exyz'])
    forces1 = np.concatenate(df['forces1_exyz'])

    assert not np.any(np.isclose(
        a=forces0,
        b=forces1,
        atol=atol,
        rtol=rtol
    )), 'Forces should be different'

    # Compare to reference
    atol = 1e-4  # should be close to reference
    rtol = 1e-3
    print("Energy")
    for f in range(4):
        print(f"Step {f}")
        print("\tPred1=Ref1".ljust(16), df['energy0_exyz'][f], '?=', ref_df['energy_ref0'][f])
        print("\tPred2=Ref2".ljust(16), df['energy1_exyz'][f], '?=', ref_df['energy_ref1'][f])
        assert np.all(np.isclose(
            a=df['energy0_exyz'][f],
            b=ref_df['energy_ref0'][f],
            atol=atol,
            rtol=rtol
        )), 'Energies should match reference0; did you compile with DDEBUG?'
        assert np.all(np.isclose(
            a=df['energy1_exyz'][f],
            b=ref_df['energy_ref1'][f],
            atol=atol,
            rtol=rtol
        )), 'Energies should match reference1; did you compile with DDEBUG?'
        assert np.all(np.isclose(
            a=df['forces0_exyz'][f],
            b=ref_df['forces_ref0'][f],
            atol=atol,
            rtol=rtol
        )), 'Forces should match reference0; did you compile with DDEBUG'
        assert np.all(np.isclose(
            a=df['forces1_exyz'][f],
            b=ref_df['forces_ref1'][f],
            atol=atol,
            rtol=rtol
        )), 'Forces should match reference1; did you compile with DDEBUG'


def test_average_single_species(tmp_path):
    test_folder = 'gpumd/dump_observer/carbon_average'
    files = [
        'C_2022_NEP3.txt',
        'C_2022_NEP3_MODIFIED.txt',
        'model.xyz',
        'reference_observer0.xyz',
        'reference_observer1.xyz',
        'run.in'
    ]
    _copy_files(files, test_folder, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    df, ref_df = _load_observer_files(tmp_path, mode='average')
    atol = 1e-8  # Anything smaller is considered ~0
    rtol = 1e-7
    assert np.all(np.isclose(
        a=df['energy0_exyz'],
        b=df['energy0_thermo'],
        atol=atol,
        rtol=rtol
    )), 'Energies should be consistent between exyz and thermo'

    # Compare to reference

    energy = np.vstack([ref_df['energy_ref0'], ref_df['energy_ref1']]).mean(axis=0)
    forces = np.vstack([ref_df['forces_ref0'], ref_df['forces_ref1']]).mean(axis=0)
    atol = 1e-4  # should be close to reference
    rtol = 1e-3
    print("Energy")
    for f in range(4):
        print(f"Step {f}")
        print("\tPred=Ref".ljust(16), df['energy0_exyz'][f], '?=', energy[f])
        print("\tPred1, pred2".ljust(16), ref_df['energy_ref0'][f], ', ', ref_df['energy_ref1'][f])
        assert np.all(np.isclose(
            a=df['energy0_exyz'][f],
            b=energy[f],
            atol=atol,
            rtol=rtol
        )), 'Energies should match reference; did you compile with DDEBUG?'
        assert np.all(np.isclose(
            a=df['forces0_exyz'][f],
            b=forces[f],
            atol=atol,
            rtol=rtol
        )), 'Forces should match reference; did you compile with DDEBUG?'


def test_species_order(tmp_path):
    '''Trigger check in gpumd if atom species are in a different order'''
    test_folder = 'gpumd/dump_observer/PbTe_species'
    files = [
        'model.xyz',
        'run.in',
        'PbTe.txt',
        'PbTe_modified.txt'
    ]
    _copy_files(files, test_folder, tmp_path)
    out = run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=False, capture_output=True)
    print(str(out.stderr))
    assert 'The atomic species and/or the order of the species are not consistent between the multiple potential' in str(out.stderr)
    assert out.returncode == 1
