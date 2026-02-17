"""
Basic Command Line Interface, provides command line controls for training, test, and inference. Be sure to import this file before `import torch`, otherwise the OMP_NUM_THREADS would not work.
"""

import os

os.environ["OMP_NUM_THREADS"] = str(2)  # limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ["MKL_NUM_THREADS"] = str(2)

from typing import *

import torch
import warnings
from model.utils import MyRichProgressBar as RichProgressBar
# from pytorch_lightning.loggers import TensorBoardLogger
from model.utils.my_logger import MyLogger as TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler

from pytorch_lightning.callbacks import (LearningRateMonitor, ModelSummary)
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning.utilities.rank_zero import rank_zero_info

torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.


class BaseCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        self.add_model_invariant_arguments_to_parser(parser)

    def add_model_invariant_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # RichProgressBar
        # parser.add_lightning_class_args(RichProgressBar, nested_key='progress_bar')
        # parser.set_defaults({"progress_bar.console_kwargs": {
        #     "force_terminal": True,
        #     "no_color": True,
        #     "width": 200,
        # }})

        # LearningRateMonitor
        parser.add_lightning_class_args(LearningRateMonitor, "learning_rate_monitor")
        learning_rate_monitor_defaults = {
            "learning_rate_monitor.logging_interval": "step",
        }
        parser.set_defaults(learning_rate_monitor_defaults)

        # ModelSummary
        parser.add_lightning_class_args(ModelSummary, 'model_summary')
        model_summary_defaults = {
            "model_summary.max_depth": 2,
        }
        parser.set_defaults(model_summary_defaults)

    def before_fit(self):
        #profiler = AdvancedProfiler(dirpath="/nvmework3/shaonian/MelSpatialNet/MelSpatialNet/logs/OnlineSpatialNet/4xSPB_Hid96_offline/FLOPs_25.5G/Training_efficiency", filename="profile.txt")
        #self.trainer.profiler = profiler
        resume_from_checkpoint: str = self.config['fit']['ckpt_path']
        if resume_from_checkpoint is not None and resume_from_checkpoint.endswith('last.ckpt'):
            # log in same dir
            # resume_from_checkpoint example: /mnt/home/quancs/projects/NBSS_pmt/logs/NBSS_ifp/version_29/checkpoints/last.ckpt
            resume_from_checkpoint = os.path.normpath(resume_from_checkpoint)
            splits = resume_from_checkpoint.split(os.path.sep)
            version = int(splits[-3].replace('version_', ''))
            save_dir = os.path.sep.join(splits[:-3])
            self.trainer.logger = TensorBoardLogger(save_dir=save_dir, name="", version=version, default_hp_metric=False)
        else:
            model_name = self.model.name if hasattr(self.model, 'name') else type(self.model).__name__
            self.trainer.logger = TensorBoardLogger('logs/', name=model_name, default_hp_metric=False)

    def before_test_old(self):
        if self.config['test']['ckpt_path'] != None:
            ckpt_path = self.config['test']['ckpt_path']
            print(-1, ckpt_path)

        else:
            #ckpt_path = self.config['test']['model.ckpt_path']
            # NEW (The fix)
            # 1. Try to find the standard --ckpt_path argument first
            ckpt_path = self.config.get('ckpt_path')
            print(0, ckpt_path)
            # 2. If not found, try to look inside the test config (fallback)
            if ckpt_path is None:
                print(1, ckpt_path)

                ckpt_path = self.config.get('test', {}).get('ckpt_path')

            warnings.warn(f"You should give --ckpt_path if you want to test, currently using: {ckpt_path}")
        epoch = os.path.basename(ckpt_path).split('_')[0]
        print(f"epoch: {epoch}")
        #epoch = "0"
        write_dir = os.path.dirname(os.path.dirname(ckpt_path))

        torch.set_num_threads(5)

        test_set = 'test'
        if 'test_set' in self.config['test']['data']:
            test_set = self.config['test']['data']["test_set"]
        elif 'init_args' in self.config['test']['data'] and 'test_set' in self.config['test']['data']['init_args']:
            test_set = self.config['test']['data']['init_args']["test_set"]
        exp_save_path = os.path.normpath(write_dir + '/' + epoch + '_' + test_set + '_set')

        self.copy_ckpt(exp_save_path=exp_save_path, ckpt_path=ckpt_path)

        import time
        # add 10 seconds for threads to simultaneously detect the next version
        self.trainer.logger = TensorBoardLogger(exp_save_path, name='', default_hp_metric=False)
        time.sleep(10)

    def before_test(self):
        # 1. Try to find standard ckpt_path (for full PL checkpoints)
        ckpt_path = self.config.get('ckpt_path')

        # 2. If not found, try to find arch_ckpt (for weights-only files)
        # This allows us to use weights-only files without crashing the Trainer
        if ckpt_path is None:
            # We check the 'model' config for arch_ckpt
            model_config = self.config.get('model', {})
            ckpt_path = model_config.get('arch_ckpt')

        # 3. Fallback to prevent crashes if nothing is found
        if ckpt_path is None:
            warnings.warn("No 'ckpt_path' or 'arch_ckpt' found. Logging path will be generic.")
            ckpt_path = "logs/unknown/dummy.ckpt"

            # --- Standard Logging Setup (using the path we found above) ---
        epoch = os.path.basename(ckpt_path).split('_')[0]

        write_dir = os.path.dirname(os.path.dirname(ckpt_path))
        if not write_dir or write_dir == ".":
            write_dir = "logs"

        torch.set_num_threads(5)

        test_set = 'test'
        test_config = self.config.get('test', {})
        data_config = test_config.get('data', {})

        if isinstance(data_config, dict) and 'test_set' in data_config:
            test_set = data_config["test_set"]
        elif isinstance(data_config, dict) and 'init_args' in data_config and 'test_set' in data_config['init_args']:
            test_set = data_config['init_args']["test_set"]

        exp_save_path = os.path.normpath(os.path.join(write_dir, f"{epoch}_{test_set}_set"))

        # Only copy if the file actually exists
        if os.path.exists(ckpt_path):
            self.copy_ckpt(exp_save_path=exp_save_path, ckpt_path=ckpt_path)

        import time
        self.trainer.logger = TensorBoardLogger(exp_save_path, name='', default_hp_metric=False)
        time.sleep(2)

    def after_test(self):
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for test is removed: ' + self.trainer.log_dir + '/' + f)

    def before_predict(self):
        if self.config['predict']['ckpt_path']:
            ckpt_path = self.config['predict']['ckpt_path']
        else:
            ckpt_path = self.config['predict']['model.arch_ckpt']
            warnings.warn(f"You are not using lightning checkpoint in prediction, currently using: {ckpt_path}")
        try:
            exp_save_path = self.config['predict']["model.output_path"]
        except:
            exp_save_path = os.path.dirname(ckpt_path) + "/" + ckpt_path.split("/")[-1].split(".")[0] + "_inference_result"
        os.makedirs(exp_save_path, exist_ok=True)
        rank_zero_info(f"saving results to: {exp_save_path}")

        import time
        # add 10 seconds for threads to simultaneously detect the next version
        self.trainer.logger = TensorBoardLogger(exp_save_path, name='', default_hp_metric=False)
        time.sleep(10)

    def after_predict(self):
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for predict is removed: ' + self.trainer.log_dir + '/' + f)

    def copy_ckpt_old(self, exp_save_path: str, ckpt_path: str):
        # copy checkpoint to save path
        from pathlib import Path
        os.makedirs(exp_save_path, exist_ok=True)
        if (Path(exp_save_path) / Path(ckpt_path).parent).exists() == False:
            import shutil
            shutil.copyfile(ckpt_path, Path(exp_save_path) / Path(ckpt_path).parent)

    def copy_ckpt(self, exp_save_path: str, ckpt_path: str):
        # Ensure the destination folder exists
        os.makedirs(exp_save_path, exist_ok=True)

        # Simple copy: take file from A, put in folder B
        import shutil
        try:
            shutil.copy(ckpt_path, exp_save_path)
        except Exception as e:
            print(f"Warning: Could not copy checkpoint: {e}")
