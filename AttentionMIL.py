import os
import sys
import random

import torch
import numpy as np
from tqdm import tqdm
from utils import utils
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from models.model import Attention
from utils.dataloader import Dataset
from utils.earlystopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from utils.utils import save_dict, metrics, plot_roc_curve
from utils.utils import print_color, print_title, print_mix, print_title_


class AttentionMIL(object):
    def __init__(self,
                 cfg: dict) -> None:
        self.cfg     = cfg
        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.writer  = SummaryWriter(log_dir=cfg['tensorboard_dir'])
        self.model   = Attention(cfg['model']).to(self.device)

    def init(self,
             dataloader: torch.utils.data.DataLoader) -> None:
        """
        Initializing the loss, optimzer, scheduler, and early stopping for
        training the model.
        """
        if self.cfg['use_weighted_loss']:
            print(f"Using weight loss with weights of {dataloader.dataset.ratio}.")
            # print_mix(f"Using weight loss with weights of __{dataloader.dataset.ratio}__.", 'RED')
            weights = torch.FloatTensor(dataloader.dataset.ratio).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        optimizer = getattr(torch.optim, self.cfg['optimizer'])
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=self.cfg['lr'],
                                   weight_decay=self.cfg['wd'])
        self.early_stopping = EarlyStopping(patience=self.cfg['patience'],
                                            lr_patience=self.cfg['lr_patience'])
        if self.cfg['use_schedular']:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=1,
                                                             gamma=0.8)

    def optimize_parameters(self,
                            loss: torch.Tensor) -> None:
        """
        Optimizing the network's parameters based on the loss value.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self,
                data: torch.Tensor) -> torch.Tensor | torch.Tensor | torch.Tensor:
        """
        Calculating the forward pass.
        """
        attention, output = self.model.forward(data)
        prob = torch.softmax(output, dim=1)
        return attention, prob, output

    def train_one_epoch(self,
                        train_dataloader: torch.utils.data.DataLoader,
                        epoch: int) -> None:
        """
        Training the model for a epoch.
        Accuracy and loss will be printed at the end of the epoch.
        """
        loss_ = 0
        gt_labels   = []
        pred_labels = []
        self.model.train()
        prefix = f'Training Epoch {epoch}: '
        for data in tqdm(train_dataloader, desc=prefix,
                dynamic_ncols=True, leave=True, position=0):
            data, label, id, padd, coords = data
            data  = data.cuda() if torch.cuda.is_available() else data
            label = label.cuda() if torch.cuda.is_available() else label
            attention, prob, predicted = self.forward(data)
            loss = self.criterion(predicted.type(torch.float), label.type(torch.long))
            self.optimize_parameters(loss)
            loss_ += loss.item() * label.shape[0]
            gt_labels   += label.cpu().numpy().tolist()
            pred_labels += torch.argmax(prob, dim=1).cpu().numpy().tolist()
        train_acc  = accuracy_score(gt_labels, pred_labels)
        train_loss = loss_ / len(gt_labels)
        self.writer.add_scalar('train/train_loss', train_loss, global_step=epoch)
        self.writer.add_scalar('train/train_acc', train_acc, global_step=epoch)
        # print_mix(f"\nEpoch __{epoch}__:", 'GREEN')
        # print_mix(f"Training accuracy and loss are __{train_acc*100:.2f}%__ and __{train_loss:.4f}__, respectively.", 'GREEN')
        print(f"\nEpoch {epoch}:")
        print(f"Training accuracy and loss are {train_acc*100:.2f}% and {train_loss:.4f}, respectively.")

    def validate(self,
                 dataloader: torch.utils.data.DataLoader,
                 epoch: int = None,
                 test: bool = False) -> None:
        """
        Validating the trained model by computing AUC, accuracy, balanced
        accuracy, and loss values.
        """
        loss_ = 0
        # patch level
        info = {'gt_label': [], 'prediction': [],
                'probability': np.array([]).reshape(0, self.cfg['num_classes']),
                'coords': [], 'attention': [], 'id': []}
        self.model.eval()
        txt = f'Validation Epoch {epoch}: ' if not test else 'Test : '
        with torch.no_grad():
            prefix = txt
            for data in tqdm(dataloader, desc=prefix,
                    dynamic_ncols=True, leave=True, position=0):

                data, label, slide_id, padd, coords = data
                data  = data.cuda() if torch.cuda.is_available() else data
                label = label.cuda() if torch.cuda.is_available() else label
                attention, prob, predicted = self.forward(data)
                if not test:
                    loss = self.criterion(predicted.type(torch.float), label.type(torch.long))
                    loss_ += loss.item() * label.shape[0]
                info['gt_label']   += label.cpu().numpy().tolist()
                info['prediction'] += torch.argmax(prob, dim=1).cpu().numpy().tolist()
                info['probability'] = np.vstack((info['probability'],
                                                 prob.cpu().numpy()))
                info['coords'] += coords
                info['id'] += slide_id
                info['attention'] += attention.cpu().numpy().tolist()

        # metris of both slide and patch level
        perf = metrics(info, self.cfg['num_classes'])
        info_ = {'slide': info, 'performance': perf}
        # If in validation mode, shows the AUC and ACC
        if not test:
            val_loss = loss_ / len(info['gt_label'])
            val_auc  = perf['overall_auc']
            val_acc  = perf['overall_acc']
            val_bacc = perf['balanced_acc']
            self.writer.add_scalar(f'validation/val_acc', val_acc, global_step=epoch)
            self.writer.add_scalar(f'validation/val_auc', val_auc, global_step=epoch)
            self.writer.add_scalar(f'validation/val_bacc', val_bacc, global_step=epoch)
            self.writer.add_scalar(f'validation/val_loss', val_loss, global_step=epoch)
            # print_mix(f"Validation AUC is __{val_auc:.4f}__, accuracy is __{val_acc*100:.2f}__, "
            #       f", balanced accuracy is __{val_bacc*100:.2f}__, and loss is __{val_loss:.4f}__.", 'GREEN')
            print(f"Validation AUC is {val_auc:.4f}, accuracy is {val_acc*100:.2f}, "
                  f", balanced accuracy is {val_bacc*100:.2f}, and loss is {val_loss:.4f}.")
            self.early_stopping(val_loss)
        return info_


    def train(self) -> None:
        # print_title('Training AttentionMIL')
        # print_mix(f"Training with __{(self.device).upper()}__ for __{self.cfg['epochs']}__ epochs.", 'RED')
        print_title_('Training AttentionMIL')
        print(f"Training with {(self.device).upper()} for {self.cfg['epochs']} epochs.")
        self.model.trainable_parameters()

        train_dataloader = Dataset(self.cfg['dataset'], state='train').run()
        valid_dataloader = Dataset(self.cfg['dataset'], state='validation').run()
        self.init(train_dataloader)

        best_valid_ = {}
        for criteria_ in self.cfg['criteria']:
            best_valid_[criteria_] = -np.inf

        for epoch in range(self.cfg['epochs']):
            self.train_one_epoch(train_dataloader, epoch)
            info = self.validate(valid_dataloader, epoch)
            perf = info['performance']
            # check if in each method there are improvements based on both auc and acc
            # then save the model with this format model_{criteria_}.pt
            for criteria, value in best_valid_.items():
                if perf[criteria] > value:
                    # save the model weights
                    best_valid_[criteria] = perf[criteria]
                    torch.save({'model': self.model.state_dict()},
                               os.path.join(self.cfg["checkpoints"], f"model_{criteria}.pth"))
                    # print_mix(f"Saved model weights for __{criteria}__ at epoch __{epoch}__.", 'RED')
                    print(f"Saved model weights for {criteria} at epoch {epoch}.")
                    save_dict(info, f"{self.cfg['dict_dir']}/validation_{criteria}.pkl")

            if self.early_stopping.early_stop:
                # print_color("\nTraining has stopped because of early stopping!", color='RED')
                print("\nTraining has stopped because of early stopping!")
                break
            # in validation, model finds out the lr needs to be reduced.
            if self.cfg['use_schedular'] and self.early_stopping.reduce_lr:
                before_lr = self.scheduler.get_last_lr()
                self.scheduler.step()
                after_lr = self.scheduler.get_last_lr()
                # print_mix(f"\nLearning rate is decreased from __{before_lr[0]}__ to __{after_lr[0]}__!", color='RED')
                print(f"\nLearning rate is decreased from {before_lr[0]} to {after_lr[0]}!")
        print("\nTraining has finished.")

    def test(self,
             use_external: bool = False) -> None:
        """
        Testing the model by printing all the metrics for each saved model in
        the slide level.
        """
        if not use_external:
            # print_title('Testing AttentionMIL')
            print_title_('Testing AttentionMIL')
            test_dataloader = Dataset(self.cfg['dataset'], state='test').run()
        else:
            # print_title(f"Testing AttentionMIL on external dataset ({self.cfg['external_test_name']}).")
            print_title_(f"Testing AttentionMIL on external dataset ({self.cfg['external_test_name']}).")
            test_dataloader = Dataset(self.cfg['dataset'], state='external').run()

        output = '||Dataset||'
        for s in self.cfg['CategoryEnum']:
            output += f'{s.name} Accuracy||'
        output += 'Weighted Accuracy||Kappa||F1 Score||AUC||Average Accuracy||\n'

        output_patch = ''
        output_slide = ''

        conf_mtrs = {}

        for criteria_ in self.cfg['criteria']:

            path = os.path.join(self.cfg["checkpoints"], f"model_{criteria_}.pth")
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state['model'], strict=True)

            name = criteria_ if not use_external else f"{criteria_}_external_{self.cfg['external_test_name']}"
            info = self.validate(test_dataloader, test=True)
            test_perf = info['performance']
            save_dict(info, f"{self.cfg['dict_dir']}/test_{name}.pkl")

            slide_metrics_ = test_perf

            output_slide += f'|{criteria_}|'

            for i in range(self.cfg['num_classes']):
                output_slide += f"{slide_metrics_['acc_per_subtype'][i]:.2f}%|"
            output_slide += f"{slide_metrics_['overall_acc']*100:.2f}%|{slide_metrics_['overall_kappa']:.4f}|{slide_metrics_['overall_f1']:.4f}|{slide_metrics_['overall_auc']:.4f}|{slide_metrics_['acc_per_subtype'].mean():.2f}%|\n"

            conf_mtrs[criteria_] = slide_metrics_['conf_mat']

            roc_path = os.path.join(self.cfg["roc_dir"], f"{name}.png")
            plot_roc_curve(slide_metrics_['roc_curve'],
                        self.cfg['num_classes'], roc_path, test_dataloader)


        print(output + output_slide)
        print("\nTesting has finished.")

    def run(self) -> None:
        if not self.cfg['only_test'] and not self.cfg['only_external_test']:
            self.train()
        if not self.cfg['only_external_test']:
            self.test()
        if self.cfg['test_external']:
            self.test(use_external=True)
