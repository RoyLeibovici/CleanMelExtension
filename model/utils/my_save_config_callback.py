from typing import List, Optional
from jsonargparse import Namespace
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.cli import SaveConfigCallback


class MySaveConfigCallback(SaveConfigCallback):
    # we dont save the asr_model in the config: saving it leads to unexpected bugs,
    # and the asr_model is only used for inference; there is no need to save the architecture.
    ignores: List[str] = ['progress_bar', 'learning_rate_monitor', 'model_summary', 'model.asr_model'] # 'model_checkpoint', 

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        for ignore in MySaveConfigCallback.ignores:
            if ignore not in self.config:
                continue
            self.del_config(ignore)
        super().setup(trainer, pl_module, stage)

    @staticmethod
    def add_ignores(ignore: str):
        MySaveConfigCallback.ignores.append(ignore)

    def del_config(self, ignore: str):
        if '.' not in ignore:
            if ignore in self.config:
                del self.config[ignore]
        else:
            config: Namespace = self.config
            ignore_namespace = ignore.split('.')
            for idx, name in enumerate(ignore_namespace):
                if idx != len(ignore_namespace) - 1:
                    if name in config:
                        config = config[name]
                    else:
                        return
                else:
                    del config[name]
