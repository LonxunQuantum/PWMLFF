"""
    module for Graphic Neural Network 

"""

from cgi import test
import cmd
import os
import re
from select import select
import sys
from tkinter import EXCEPTION
from turtle import title
from unittest.mock import NonCallableMagicMock
from os.path import isdir
from pathlib import Path
import yaml

# for data generation 
import numpy as np
import pymatgen as pym

codepath = str(Path(__file__).parent.resolve())

sys.path.append(codepath+'/../aux')

# for training 
import torch

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.data import dataset_from_config
from nequip.utils import load_file
from nequip.utils.test import assert_AtomicData_equivariant
from nequip.utils.versions import check_code_version,get_config_code_versions
from nequip.utils._global_options import _set_global_options
from nequip.scripts._logger import set_up_script_logger

# for evaluation 
import argparse
import textwrap
from pathlib import Path  
import logging
from nequip.scripts.deploy import load_deployed_model, R_MAX_KEY
from nequip.scripts.train import default_config, check_code_version
from nequip.train import Trainer, Loss, Metrics
from nequip.data import AtomicData, Collater, dataset_from_config, register_fields
from nequip.utils import load_file, instantiate
import contextlib
from tqdm.auto import tqdm

#  for depolying 
from nequip.scripts.deploy import _compile_for_deploy

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final



class gnn_network:
    
    def __init__(   self,

                    #job related                 
                    session_dir = "record",
                    task_name = "gnn",
                    seed = 123,
                    dataset_seed = 3581,
                    append = False, 
                    default_dtype = "float64",
                    device = "cuda", 

                    # network related

                    # cutoff radius in length units, here Angstrom, this is an important hyperparamter to scan
                    r_max = 4.0,                                                     
                    
                    # number of interaction blocks, we find 3-5 to work best
                    num_layers = 4, 

                    # the maximum irrep order (rotation order) for the network's features, l=1 is a good default, l=2 is more accurate but slower
                    l_max = 1, 

                    # whether to include features with odd mirror parityy; often turning parity off gives equally good results but faster networks, so do consider this
                    parity = True,
                    
                    # the multiplicity of the features, 32 is a good default for accurate network, if you want to be more accurate, go larger, if you want to be faster, go lower
                    num_features = 32, 
                    nonlinearity_type = 'gate',

                    nonlinearity_scalars = {"e": "silu" , "o": "tanh"}, 
                    nonlinearity_gates = {"e": "silu" , "o": "tanh"},
                    
                    # radial network 

                    # number of basis functions used in the radial basis, 8 usually works best
                    num_basis = 8,         

                    # set true to train the bessel weights                                                  
                    BesselBasis_trainable = True,     

                    # p-exponent used in polynomial cutoff function, smaller p corresponds to stronger decay with distance
                    PolynomialCutoff_p = 6,                 

                    # number of radial layers, usually 1-3 works best, smaller is faster
                    invariant_layers = 2,         

                    # number of hidden neurons in radial function, smaller is faster
                    invariant_neurons = 64,                                         

                    # number of neighbors to divide by, null => no normalization, auto computes it based on dataset
                    avg_num_neighbors = "auto",  

                    # use self-connection or not, usually gives big improvement
                    use_sc = True,                                                                     

                    # dataset  
                    dataset = "ase",
                    dataset_file_name = r"PWdata/training_data.xyz",
                    ase_args = {"format":"extxyz"},
                    chemical_symbols = ["Cu"], 

                    # info 
                    verbose = "info",                                                                    # the same as python logging, e.g. warning, info, debug, error; case insensitive
                    log_batch_freq = 1,                                                                  # batch frequency, how often to print training errors withinin the same epoch
                    log_epoch_freq = 1,                                                                  # epoch frequency, how often to print
                    save_checkpoint_freq = -1,                                                           # frequency to save the intermediate checkpoint. no saving of intermediate checkpoints when the value is not positive.
                    save_ema_checkpoint_freq = -1,
                    
                    # training 
                    num_train_img = 600,                                                                       # number of training data
                    num_valid_img = 200,                                                                         # number of validation data
                    learning_rate = 0.005,                                                               # learning rate, we found values between 0.01 and 0.005 to work best - this is often one of the most important hyperparameters to tune
                    train_batch_size = 5,                                                                      # batch size, we found it important to keep this small for most applications including forces (1-5); for energy-only training, higher batch sizes work better
                    valid_batch_size = 10,                                                          # batch size for evaluating the model during validation. This does not affect the training results, but using the highest value possible (<=n_val) without running out of memory will speed up your training.
                    epoch_num = 10,                                                                     # stop training after _ number of epochs, we set a very large number, as e.g. 1million and then just use early stopping and not train the full number of epochs
                    train_val_split = "random",                                                            # can be random or sequential. if sequential, first n_train elements are training, next n_val are val, else random, usually random is the right choice
                    shuffle = True,                                                                       # if true, the data loader will shuffle the data, usually a good idea
                    metrics_key = "validation_loss",                                                       # metrics used for scheduling and saving best model. Options: `set`_`quantity`, set can be either "train" or "validation, "quantity" can be loss or anything that appears in the validation batch step header, such as f_mae, f_rmse, e_mae, e_rmse
                    use_ema = True,                                                                       # if true, use exponential moving average on weights for val/test, usually helps a lot with training, in particular for energy errors
                    ema_decay = 0.99,                                                                     # ema weight, typically set to 0.99 or 0.999
                    ema_use_num_updates = True,                                                           # whether to use number of updates when computing averages
                    report_init_validation = True,                                                        # if True, report the validation error for just initialized model       
                    
                    # early stoping 
                    early_stopping_patiences =  {'validation_loss': 50}, 
                    early_stopping_lower_bounds = {'LR': 1e-05}, 

                    # loss function 
                    loss_coeffs = {'forces': 1, 'total_energy': [3, 'MSELoss']},

                    # optimizer 
                    optimizer_name = "Adam",
                    optimizer_amsgrad = False, 
                    optimizer_betas = (0.9,0.999),         #   python/tuple
                    optimizer_eps = 1.0e-08,
                    optimizer_weight_decay = 0,

                    # scheduler 
                    lr_scheduler_name = "ReduceLROnPlateau",
                    lr_scheduler_patience = 100, 
                    lr_scheduler_factor = 0.5, 

                    # scaling 
                    per_species_rescale_shifts_trainable = False,
                    per_species_rescale_scales_trainable = False,
                    per_species_rescale_shifts = "dataset_per_atom_total_energy_mean",
                    per_species_rescale_scales = "dataset_forces_rms"
                    
                    # non-original parameters 
                    
                    ):
        """
            yank nequip's default config 
        """ 
        input_args = locals()  
        del input_args["self"]
        
        """
            change 
        """
        
        """
            Dump the input_args if needed. 
        """ 

        config_raw = dict(
                wandb=False,
                wandb_project="NequIP",
                model_builders=[
                    "SimpleIrrepsConfig",
                    "EnergyModel",
                    "PerSpeciesRescale",
                    "ForceOutput",
                    "RescaleEnergyEtc",
                ],
                dataset_statistics_stride=1,

                allow_tf32=False,  # TODO: until we understand equivar issues

                model_debug_mode=False,
                equivariance_test=False,
                grad_anomaly_mode=False,
                _jit_bailout_depth=2, 
                _jit_fusion_strategy=[("DYNAMIC", 3)],
            )
        
        # add all args to the config
        for args in input_args:
            # some re-parsing here for user's delight

            if args == "session_dir":
                config_raw["root"] = input_args[args]
            
            elif args == "task_name": 
                config_raw["run_name"] = input_args[args]

            elif args == "epoch_num":
                config_raw["max_epochs"] = input_args[args]

            elif args == "num_train_img":
                config_raw["n_train"] = input_args[args]
            
            elif args == "num_valid_img":
                config_raw["n_val"] = input_args[args]

            elif args == "train_batch_size":
                config_raw["batch_size"] = input_args[args]

            elif args == "valid_batch_size":
                config_raw["validation_batch_size"] = input_args[args]

            else:
                config_raw[args] = input_args[args]
        
        #print("CONFIGURATION OF TRAINER:\n" )  
        
        """
            add user-defined ones 
        """ 
        
        # NEquiP's Config class. All in one 
        self.config = Config(config_raw)
        
        print ("************************************************************")
        print ("*    Neural Equivariant Interatomic Potentials (NequIP)    *")
        print ("*     An E(3)-equivariant neural network approach for      *")
        print ("*     learning interatomic potentials from ab-initio       *")
        print ("*     calculations for molecular dynamics simulations.     *") 
        print ("*    Authors: Batzner, S., Musaelian, A., Sun, L. et al.   *")
        print ("*    DOI: https://doi.org/10.1038/s41467-022-29939-5       *")
        print ("*    GitHub: https://github.com/mir-group/nequip           *")
        print ("************************************************************")
        
    """
        Auxiliary functions 
    """ 
    def set_device(self,val):
        self.config["device"] = val
    
    def set_session_dir(self,val):
        self.config["root"] = val 

    def set_task_name(self,val):
        self.config["run_name"] = val
    
    def set_epoch_num(self,val):
        self.config["max_epochs"] = val

    def set_num_train_img(self,val):
        self.config["n_train"] = val
    
    def set_num_valid_img(self,val):
        self.config["n_val"] = val 
    
    def set_train_batch_size(self,val):
        self.config["batch_size"] = val 
    
    def set_valid_batch_size(self,val):
        self.config["validation_batch_size"] = val

    def set_learning_rate(self,val):
        self.config["learning_rate"] = val 
    
    # network related 
    def set_r_max(self,val):
        # cutoff radius in length units, here Angstrom, this is an important hyperparamter to scan
        self.config["r_max"] = val
    
    def set_num_layers(self,val):
        # number of interaction blocks, we find 3-5 to work best
        self.config["num_layers"] = val 
    
    def set_l_max(self,val):
        # the maximum irrep order (rotation order) for the network's features, l=1 is a good default, l=2 is more accurate but slower
        self.config["l_max"] = val

    def set_num_features(self,val):
        # the multiplicity of the features, 32 is a good default for accurate network, if you want to be more accurate, go larger, if you want to be faster, go lower
        self.config["num_features"] = val

    def set_num_basis(self,val):
        # number of basis functions used in the radial basis, 8 usually works best
        self.config["num_basis"] = val 

    def set_PolynomialCutoff_p(self,val):
        # p-exponent used in polynomial cutoff function, smaller p corresponds to stronger decay with distance
        self.config["PolynomialCutoff_p"] = val 

    def set_invariant_layers(self,val):
        # number of radial layers, usually 1-3 works best, smaller is faster
        self.config["invariant_layers"] = val 
    
    """
        Generating data
    """

    def generate_data(self,xyz_output = "./PWdata/training_data.xyz"):
        """
            convert MOVEMENT to .xyz
            multiple MOVEMENTs are concatonate into MOVEMENT_ALL
            default output: PWdata/training_data.xyz
        """    
        
        # the converting module 
        import gnn_convert
        import subprocess
        
        # find dirs that contain MOVEMENT
        mvmt_dir = [] 
        tmpxyz_dir = []

        # a single MOVEMENT in PWdata
        if os.path.exists("PWdata/MOVEMENT"):
            mvmt_dir.append("PWdata/MOVEMENT")
            tmpxyz_dir.append("PWdata/tmp.xyz")
        # multiple movements in PWdata
        else:
            mvmt_dir = ["PWdata/"+name+"/MOVEMENT" for name in os.listdir("./PWdata") if os.path.isdir("./PWdata/"+name)]
            tmpxyz_dir = ["PWdata/"+name+"/tmp.xyz" for name in os.listdir("./PWdata") if os.path.isdir("./PWdata/"+name)]
        
        print("These files will be used for trianing:")
        for dir in mvmt_dir:
            #read_name = r"./PWdata/"+name+"/MOVEMENT"    
            print (dir)
        
        if len(mvmt_dir)==0:
            raise Exception("Input MOVEMENT for training not found")

        total_img_num = 0 

        # concatenate .xyz instead of movement 

        for mvmt_name, xyz_name in zip(mvmt_dir,tmpxyz_dir):
            print ("\nprocessing "+mvmt_name)

            a= gnn_convert.Structure(mvmt_name,type='MOVEMENT')

            a.coordinate2cartesian()

            img_tmp = a.out_extxyz(xyz_name)  
            total_img_num += img_tmp
        
        cmd_cat = "cat "

        for xyz_name in tmpxyz_dir:
            cmd_cat = cmd_cat + xyz_name + " "
        
        cmd_cat = cmd_cat + "> " + xyz_output
        
        print (cmd_cat)
        subprocess.run(cmd_cat,shell=True)

        print(".xyz file has been saved to:"+xyz_output)
        print("total number of image:",total_img_num)


    def train(self, train_data = r"./PWdata/training_data.xyz" ):
        
        """
            Launching training 
        """
        # update training data path  
        self.config["dataset_file_name"]  = train_data
        
        if not os.path.exists(self.config["dataset_file_name"]):
            raise Exception(train_data,"not found")

        from nequip.scripts.train import restart, fresh_start

        set_up_script_logger(self.config.get("log", None), self.config.verbose)

        found_restart_file = isdir(f"{self.config.root}/{self.config.run_name}")

        if found_restart_file and not self.config.append:
            raise RuntimeError(
                f"Training instance exists at {self.config.root}/{self.config.run_name}; "
                "either set append to True or use a different root or runname"
            )
        if not found_restart_file:
            trainer = fresh_start(self.config)
        else:
            trainer = restart(self.config)
        
        trainer.save()
        trainer.train()
        
        return 
        
    def evaluate(   self, 
                    # must specify     
                    train_dir = None,
                    model = None,   
                    metrics_config = None, 
                    test_indices = None,
                    batch_size = 50, 
                    repeat = 1, 
                    device = None, 
                    use_deterministic_algorithms = False, 
                    output = None, 
                    output_fields = "", 
                    log = None
                ):
        """
            evaluate the trained model 
            arg values are passed in inside the python 
            refer to NequiP manual for more information 
        """
        ORIGINAL_DATASET_INDEX_KEY: str = "original_dataset_index"
        register_fields(graph_fields=[ORIGINAL_DATASET_INDEX_KEY])

        parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Compute the error of a model on a test set using various metrics.

            The model, metrics, dataset, etc. can specified individually, or a training session can be indicated with `--train-dir`.
            In order of priority, the global settings (dtype, TensorFloat32, etc.) are taken from:
              1. The model config (for a training session)
              2. The dataset config (for a deployed model)
              3. The defaults

            Prints only the final result in `name = num` format to stdout; all other information is logging.debuged to stderr.

            WARNING: Please note that results of CUDA models are rarely exactly reproducible, and that even CPU models can be nondeterministic.
            """
            )
        )
        parser.add_argument(
            "--train-dir",
            help="Path to a working directory from a training session.",
            type=Path,
            default=None,
        )
        parser.add_argument(
            "--model",
            help="A deployed or pickled NequIP model to load. If omitted, defaults to `best_model.pth` in `train_dir`.",
            type=Path,
            default=None,
        )
        parser.add_argument(
            "--dataset-config",
            help="A YAML config file specifying the dataset to load test data from. If omitted, `config.yaml` in `train_dir` will be used",
            type=Path,
            default=None,
        )
        parser.add_argument(
            "--metrics-config",
            help="A YAML config file specifying the metrics to compute. If omitted, `config.yaml` in `train_dir` will be used. If the config does not specify `metrics_components`, the default is to logging.debug MAEs and RMSEs for all fields given in the loss function. If the literal string `None`, no metrics will be computed.",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--test-indexes",
            help="Path to a file containing the indexes in the dataset that make up the test set. If omitted, all data frames *not* used as training or validation data in the training session `train_dir` will be used.",
            type=Path,
            default=None,
        )
        parser.add_argument(
            "--batch-size",
            help="Batch size to use. Larger is usually faster on GPU. If you run out of memory, lower this.",
            type=int,
            default=50,
        )
        parser.add_argument(
            "--repeat",
            help=(
                "Number of times to repeat evaluating the test dataset. "
                "This can help compensate for CUDA nondeterminism, or can be used to evaluate error on models whose inference passes are intentionally nondeterministic. "
                "Note that `--repeat`ed passes over the dataset will also be `--output`ed if an `--output` is specified."
            ),
            type=int,
            default=1,
        )
        parser.add_argument(
            "--use-deterministic-algorithms",
            help="Try to have PyTorch use deterministic algorithms. Will probably fail on GPU/CUDA.",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--device",
            help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--output",
            help="ExtXYZ (.xyz) file to write out the test set and model predictions to.",
            type=Path,
            default=None,
        )
        parser.add_argument(
            "--output-fields",
            help="Extra fields (names comma separated with no spaces) to write to the `--output`.",
            type=str,
            default="",
        )
        parser.add_argument(
            "--log",
            help="log file to store all the metrics and screen logging.debug",
            type=Path,
            default=None,
        )
        
        # create a n args container 
        args = parser.parse_args(args=None)
        
        # fill the values
        args.batch_size = batch_size
        #args.dataset_config = 
        args.device = device 

        if log is not None:
            args.log = Path(log)

        args.metrics_config = metrics_config 

        if model is not None:
            args.model = Path(model)
        else:
            raise Exception("must specify the model to be evaluated")
            

        if output is not None:
            args.output = Path(output)

        args.output_fields = output_fields

        args.repeat = repeat

        if test_indices is not None:
            args.test_indexes = Path(test_indices)

        if train_dir is not None:
            args.train_dir = Path(train_dir)
        else:
            raise Exception("a training directory must be specified")
        
        args.use_deterministic_algorithms = use_deterministic_algorithms
        
        # Do the defaults:
        dataset_is_from_training: bool = False
        if args.train_dir:
            if args.dataset_config is None:
                args.dataset_config = args.train_dir / "config.yaml"
                dataset_is_from_training = True
            if args.metrics_config is None:
                args.metrics_config = args.train_dir / "config.yaml"
            if args.model is None:
                args.model = args.train_dir / "best_model.pth"
            if args.test_indexes is None:
                # Find the remaining indexes that arent train or val
                trainer = torch.load(
                    str(args.train_dir / "trainer.pth"), map_location="cpu"
                )
                train_idcs = set(trainer["train_idcs"].tolist())
                val_idcs = set(trainer["val_idcs"].tolist())
            else:
                train_idcs = val_idcs = None
        # update
        if args.metrics_config == "None":
            args.metrics_config = None
        elif args.metrics_config is not None:
            args.metrics_config = Path(args.metrics_config)
        do_metrics = args.metrics_config is not None
        # validate
        if args.dataset_config is None:
            raise ValueError("--dataset-config or --train-dir must be provided")
        if args.metrics_config is None and args.output is None:
            raise ValueError(
                "Nothing to do! Must provide at least one of --metrics-config, --train-dir (to use training config for metrics), or --output"
            )
        if args.model is None:
            raise ValueError("--model or --train-dir must be provided")
        output_type: Optional[str] = None
        if args.output is not None:
            if args.output.suffix != ".xyz":
                raise ValueError("Only .xyz format for `--output` is supported.")
            args.output_fields = [e for e in args.output_fields.split(",") if e != ""] + [
                ORIGINAL_DATASET_INDEX_KEY
            ]
            output_type = "xyz"
        else:
            assert args.output_fields == ""
            args.output_fields = []

        if args.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        # setting up logger 
        if True:
            set_up_script_logger(args.log)
        logger = logging.getLogger("nequip-evaluate")
        logger.setLevel(logging.INFO)

        logger.info(f"Using device: {device}")
        if device.type == "cuda":
            logger.info(
                "WARNING: please note that models running on CUDA are usually nondeterministc and that this manifests in the final test errors; for a _more_ deterministic result, please use `--device cpu`",
            )
        
        if args.use_deterministic_algorithms:
            logger.info(
                "Telling PyTorch to try to use deterministic algorithms... please note that this will likely error on CUDA/GPU"
            )
            torch.use_deterministic_algorithms(True)
        
        # Load model:
        logger.info("Loading model... ")
        loaded_deployed_model: bool = False
        model_r_max = None
        try:
            model, metadata = load_deployed_model(
                args.model,
                device=device,
                set_global_options=True,  # don't warn that setting
            )
            logger.info("loaded deployed model.")
            # the global settings for a deployed model are set by
            # set_global_options in the call to load_deployed_model
            # above
            model_r_max = float(metadata[R_MAX_KEY])
            loaded_deployed_model = True
        except ValueError:  # its not a deployed model
            loaded_deployed_model = False
        # we don't do this in the `except:` block to avoid "during handing of this exception another exception"
        # chains if there is an issue loading the training session model. This makes the error messages more
        # comprehensible:
        logger.info("Model loaded") 

        if not loaded_deployed_model:
            # Use the model config, regardless of dataset config
            global_config = args.model.parent / "config.yaml"
            global_config = Config.from_file(str(global_config), defaults=default_config)
            _set_global_options(global_config)
            check_code_version(global_config)
            del global_config

            # load a training session model
            model, model_config = Trainer.load_model_from_training_session(
                traindir=args.model.parent, model_name=args.model.name
            )
            model = model.to(device)
            logger.info("loaded model from training session")
            model_r_max = model_config["r_max"]

        model.eval()
        

        logger.info(
            f"Loading {'original ' if dataset_is_from_training else ''}dataset...",
        )
        dataset_config = Config.from_file(
            str(args.dataset_config), defaults={"r_max": model_r_max}
        )
        if dataset_config["r_max"] != model_r_max:
            raise RuntimeError(
                f"Dataset config has r_max={dataset_config['r_max']}, but model has r_max={model_r_max}!"
            )

        dataset_is_validation: bool = False
        try:
            # Try to get validation dataset
            dataset = dataset_from_config(dataset_config, prefix="validation_dataset")
            dataset_is_validation = True
        except KeyError:
            pass
        if not dataset_is_validation:
            # Get shared train + validation dataset
            # prefix `dataset`
            dataset = dataset_from_config(dataset_config)
        logger.info(
            f"Loaded {'validation_' if dataset_is_validation else ''}dataset specified in {args.dataset_config.name}.",
        )

        c = Collater.for_dataset(dataset, exclude_keys=[])

        # Determine the test set
        # this makes no sense if a dataset is given seperately
        if (
            args.test_indexes is None
            and dataset_is_from_training
            and train_idcs is not None
        ):
            # we know the train and val, get the rest
            all_idcs = set(range(len(dataset)))
            # set operations
            if dataset_is_validation:
                test_idcs = list(all_idcs - val_idcs)
                logger.info(
                    f"Using origial validation dataset ({len(dataset)} frames) minus validation set frames ({len(val_idcs)} frames), yielding a test set size of {len(test_idcs)} frames.",
                )
            else:
                test_idcs = list(all_idcs - train_idcs - val_idcs)
                assert set(test_idcs).isdisjoint(train_idcs)
                logger.info(
                    f"Using origial training dataset ({len(dataset)} frames) minus training ({len(train_idcs)} frames) and validation frames ({len(val_idcs)} frames), yielding a test set size of {len(test_idcs)} frames.",
                )
            # No matter what it should be disjoint from validation:
            assert set(test_idcs).isdisjoint(val_idcs)
            if not do_metrics:
                logger.info(
                    "WARNING: using the automatic test set ^^^ but not computing metrics, is this really what you wanted to do?",
                )
        elif args.test_indexes is None:
            # Default to all frames
            test_idcs = torch.arange(dataset.len())
            logger.info(
                f"Using all frames from the specified test dataset, yielding a test set size of {len(test_idcs)} frames.",
            )
        else:
            # load from file
            test_idcs = load_file(
                supported_formats=dict(
                    torch=["pt", "pth"], yaml=["yaml", "yml"], json=["json"]
                ),
                filename=str(args.test_indexes),
            )
            logger.info(
                f"Using provided test set indexes, yielding a test set size of {len(test_idcs)} frames.",
            )
        
        test_idcs = torch.as_tensor(test_idcs, dtype=torch.long)
        test_idcs = test_idcs.tile((args.repeat,))

        # Figure out what metrics we're actually computing
        if do_metrics:
            metrics_config = Config.from_file(str(args.metrics_config))
            metrics_components = metrics_config.get("metrics_components", None)
            # See trainer.py: init() and init_metrics()
            # Default to loss functions if no metrics specified:
            if metrics_components is None:
                loss, _ = instantiate(
                    builder=Loss,
                    prefix="loss",
                    positional_args=dict(coeffs=metrics_config.loss_coeffs),
                    all_args=metrics_config,
                )
                metrics_components = []
                for key, func in loss.funcs.items():
                    params = {
                        "PerSpecies": type(func).__name__.startswith("PerSpecies"),
                    }
                    metrics_components.append((key, "mae", params))
                    metrics_components.append((key, "rmse", params))

            metrics, _ = instantiate(
                builder=Metrics,
                prefix="metrics",
                positional_args=dict(components=metrics_components),
                all_args=metrics_config,
            )
            metrics.to(device=device)
    
        batch_i: int = 0
        batch_size: int = args.batch_size

        logger.info("Starting...")
        context_stack = contextlib.ExitStack()
        with contextlib.ExitStack() as context_stack:
            # "None" checks if in a TTY and disables if not
            prog = context_stack.enter_context(tqdm(total=len(test_idcs), disable=None))
            if do_metrics:
                display_bar = context_stack.enter_context(
                    tqdm(
                        bar_format=""
                        if prog.disable  # prog.ncols doesn't exist if disabled
                        else ("{desc:." + str(prog.ncols) + "}"),
                        disable=None,
                    )
                )

            if output_type is not None:
                output = context_stack.enter_context(open(args.output, "w"))
            else:
                output = None
        
            while True:
                this_batch_test_indexes = test_idcs[
                    batch_i * batch_size : (batch_i + 1) * batch_size
                ]
                datas = [dataset[int(idex)] for idex in this_batch_test_indexes]
                if len(datas) == 0:
                    break
                batch = c.collate(datas)
                batch = batch.to(device)
                out = model(AtomicData.to_AtomicDataDict(batch))

                with torch.no_grad():
                    # Write output
                    if output_type == "xyz":
                        # add test frame to the output:
                        out[ORIGINAL_DATASET_INDEX_KEY] = torch.LongTensor(
                            this_batch_test_indexes
                        )
                        # append to the file
                        ase.io.write(
                            output,
                            AtomicData.from_AtomicDataDict(out)
                            .to(device="cpu")
                            .to_ase(
                                type_mapper=dataset.type_mapper,
                                extra_fields=args.output_fields,
                            ),
                            format="extxyz",
                            append=True,
                        )

                    # Accumulate metrics
                    if do_metrics:
                        metrics(out, batch)
                        display_bar.set_description_str(
                            " | ".join(
                                f"{k} = {v:4.4f}"
                                for k, v in metrics.flatten_metrics(
                                    metrics.current_result(),
                                    type_names=dataset.type_mapper.type_names,
                                )[0].items()
                            )
                        )

                batch_i += 1
                prog.update(batch.num_graphs)

            prog.close()
            if do_metrics:
                display_bar.close()

        if do_metrics:
            logger.info("\n--- Final result: ---")
            logger.critical(
                "\n".join(
                    f"{k:>20s} = {v:< 20f}"
                    for k, v in metrics.flatten_metrics(
                        metrics.current_result(),
                        type_names=dataset.type_mapper.type_names,
                    )[0].items()
                )
            )

    def deploy( self,
                model = None, 
                train_dir = None, 
                out_file = None
                ):
        """
            deploying model for production 
            only command = build
        """
        
        CONFIG_KEY: Final[str] = "config"
        NEQUIP_VERSION_KEY: Final[str] = "nequip_version"
        TORCH_VERSION_KEY: Final[str] = "torch_version"
        E3NN_VERSION_KEY: Final[str] = "e3nn_version"
        CODE_COMMITS_KEY: Final[str] = "code_commits"
        R_MAX_KEY: Final[str] = "r_max"
        N_SPECIES_KEY: Final[str] = "n_species"
        TYPE_NAMES_KEY: Final[str] = "type_names"
        JIT_BAILOUT_KEY: Final[str] = "_jit_bailout_depth"
        JIT_FUSION_STRATEGY: Final[str] = "_jit_fusion_strategy"
        TF32_KEY: Final[str] = "allow_tf32"

        _ALL_METADATA_KEYS = [
            CONFIG_KEY,
            NEQUIP_VERSION_KEY,
            TORCH_VERSION_KEY,
            E3NN_VERSION_KEY,
            R_MAX_KEY,
            N_SPECIES_KEY,
            TYPE_NAMES_KEY,
            JIT_BAILOUT_KEY,
            JIT_FUSION_STRATEGY,
            TF32_KEY,
        ]

        parser = argparse.ArgumentParser(
        description="Create and view information about deployed NequIP potentials."
        )

        parser.add_argument(
            "--model",
            help="Path to a YAML file defining a model to deploy. Unless you know why you need to, do not use this option.",
            type=Path,
        )
        
        parser.add_argument(
            "--train-dir",
            help="Path to a working directory from a training session to deploy.",
            type=Path,
        )
        parser.add_argument(
            "--out_file",
            help="Output file for deployed model.",
            type=Path,
        )
        
        args = parser.parse_args(args=None)

        if model is not None:
            args.model = Path(model)

        if train_dir is not None:
            args.train_dir = Path(train_dir)
        
        if out_file is not None:
            args.out_file = Path(out_file)
        
        # start building 

        if args.model and args.train_dir:
            raise ValueError("--model and --train-dir cannot both be specified.")
        if args.train_dir is not None:
            logging.info("Loading best_model from training session...")
            config = Config.from_file(str(args.train_dir / "config.yaml"))
        elif args.model is not None:
            logging.info("Building model from config...")
            config = Config.from_file(str(args.model), defaults=default_config)
        else:
            raise ValueError("one of --train-dir or --model must be given")

        _set_global_options(config)
        check_code_version(config)

        # -- load model --
        if args.train_dir is not None:
            model, _ = Trainer.load_model_from_training_session(
                args.train_dir, model_name="best_model.pth", device="cpu"
            )
        elif args.model is not None:
            model = model_from_config(config)
        else:
            raise AssertionError

        # -- compile --
        model = _compile_for_deploy(model)
        logging.info("Compiled & optimized model.")

        # Deploy
        metadata: dict = {}
        code_versions, code_commits = get_config_code_versions(config)
        for code, version in code_versions.items():
            metadata[code + "_version"] = version
        if len(code_commits) > 0:
            metadata[CODE_COMMITS_KEY] = ";".join(
                f"{k}={v}" for k, v in code_commits.items()
            )

        metadata[R_MAX_KEY] = str(float(config["r_max"]))
        if "allowed_species" in config:
            # This is from before the atomic number updates
            n_species = len(config["allowed_species"])
            type_names = {
                type: ase.data.chemical_symbols[atomic_num]
                for type, atomic_num in enumerate(config["allowed_species"])
            }
        else:
            # The new atomic number setup
            n_species = str(config["num_types"])
            type_names = config["type_names"]
        metadata[N_SPECIES_KEY] = str(n_species)
        metadata[TYPE_NAMES_KEY] = " ".join(type_names)

        metadata[JIT_BAILOUT_KEY] = str(config[JIT_BAILOUT_KEY])
        if int(torch.__version__.split(".")[1]) >= 11 and JIT_FUSION_STRATEGY in config:
            metadata[JIT_FUSION_STRATEGY] = ";".join(
                "%s,%i" % e for e in config[JIT_FUSION_STRATEGY]
            )
        metadata[TF32_KEY] = str(int(config["allow_tf32"]))
        metadata[CONFIG_KEY] = yaml.dump(dict(config))

        metadata = {k: v.encode("ascii") for k, v in metadata.items()}
        torch.jit.save(model, args.out_file, _extra_files=metadata)

        print("model saved to:", args.out_file)
        
        return
        
        