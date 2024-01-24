from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
# from SLPT.model import Sparse_alignment_network
# from SLPT.Dataloader.ffhq_loader import ffhq_dataset
from dataset.rotate_ffhq import ffhq_dataset

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

class SaveModelAfterNEpochs(pl.callbacks.Callback):
    def __init__(self, save_epoch_interval, save_iteration_interval, save_path):
        self.save_epoch_interval = save_epoch_interval
        self.save_path = save_path
        self.save_iteration_interval = save_iteration_interval
        self.current_iteration = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.current_iteration % self.save_iteration_interval == 0:
            checkpoint_path = f"{self.save_path}/model_v16_after_iteration.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved model parameters at iteration {self.current_iteration} to {checkpoint_path}")
        self.current_iteration += 1

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.save_epoch_interval == 0:
            checkpoint_path = f"{self.save_path}/model_v16_after_epoch.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved model parameters at epoch {trainer.current_epoch} to {checkpoint_path}")


if __name__ == "__main__":
    # Configs
    resume_path = './models/control_sd21_v14_ini.ckpt'
    batch_size = 4
    logger_freq = 10000
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False


    ffhq_data = ffhq_dataset(file_list="/home/zhongtao/datasets/ffhq_rotate", out_size=512, crop_type="none")
    train_dataloader = DataLoader(ffhq_data,num_workers=8, batch_size=batch_size, shuffle=True)

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('models/cldm_v21_parsing14.yaml').cpu()
    # TODO：ID_module_Pretrained = None
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # valida_dataloader = DataLoader(ffhq_dataset, num_workers=8, batch_size=1, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)

    # 创建 ModelCheckpoint 回调

    save_callback = SaveModelAfterNEpochs(2, 4000, "./models")

    trainer = pl.Trainer(gpus=[1], precision=32, callbacks=[logger], val_check_interval=2000, log_every_n_steps=5000,
                         max_epochs=100)
    trainer.callbacks.append(save_callback)

    # Train!
    trainer.fit(model, train_dataloaders=train_dataloader)
