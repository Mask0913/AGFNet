import time

import torch
from dataset import data
from models import segmodel
from tools.compute import ConfusionMatrix
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os


class Train_model:

    def __init__(
            self,
            model,
            loss_fun,
            optimizer,
            lr_scheduler,
            data_loader,
            detail_loss,
            config):
        self.model = model
        self.seg_loss = loss_fun

        self.seg_loss = self.seg_loss.cuda()

        self.alpha = 0
        self.beta = 0
        self.gamma = 0


        self.optimizer = optimizer
        self.train_data_loader, self.test_data_loader, self.val_data_loader = data_loader
        self.iter = config.iter
        self.lr_scheduler = lr_scheduler
        self.pre_miou = 0
        if not os.path.exists(config.log_path):
            os.makedirs(config.log_path)
        self.log_path = config.log_path
        self.writer = SummaryWriter(self.log_path)
        self.class_num = config.class_num
        self.compute = ConfusionMatrix(self.class_num, [])
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.print(config)
        self.model.train()

        if config.multi_loss:
            self.detail_loss = detail_loss
            self.detail_loss = self.detail_loss.cuda()
            self.train = self.multi_loss_train
        else:
            self.detail_loss = None
            self.train = self.singleloss_train



    def multi_loss_train(self):
        i = 0
        start_time = time.time()
        while i < self.iter:
            now_time = time.time()
            for _, (image, label, index) in enumerate(self.train_data_loader):
                image = image.to(self.device)
                label = label.to(self.device)
                y1, y2, y3, boundary = self.model(image)
                loss1 = self.seg_loss(y1, label)
                loss2 = self.seg_loss(y2, label)
                loss3 = self.seg_loss(y3, label)
                loss4, loss5 = self.detail_loss(boundary, label)
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                self.alpha = self.model.UAFM_T2.rgb_scale.data
                self.beta = self.model.UAFM_T2.high_scale.data
                self.gamma = self.model.UAFM_T2.t_scale.data
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                if i % 40 == 0:
                    one_time = time.time() - now_time
                    used_time = (time.time() - start_time) / 60
                    end_time = used_time / (i + 1) * self.iter
                    self.print(
                        "iter: {}, loss: {}, time: {}s, end_time: {}min, used_time: {}min".format(
                            i, loss, round(
                                one_time, 2), round(
                                end_time, 2), round(
                                used_time, 2)))
                    now_time = time.time()
                if i % 400 == 0 and i != 0:
                    self.val_step(i)
                    now_time = time.time()
                i += 1
                if i >= self.iter:
                    break
        self.writer.close()
        print("train end")



    def singleloss_train(self):
        i = 0
        start_time = time.time()
        while i < self.iter:
            now_time = time.time()
            for _, (image, label, index) in enumerate(self.train_data_loader):
                image = image.to(self.device)
                label = label.to(self.device)
                y = self.model(image)
                loss = self.seg_loss(y, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.writer.add_scalar("loss", loss, i)
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], i)
                if i % 40 == 0:
                    one_time = time.time() - now_time
                    used_time = (time.time() - start_time) / 60
                    end_time = used_time / (i + 1) * self.iter
                    self.print(
                        "iter: {}, loss: {}, time: {}s, end_time: {}min, used_time: {}min".format(
                            i, loss, round(
                                one_time, 2), round(
                                end_time, 2), round(
                                used_time, 2)))
                    now_time = time.time()
                if i % 400 == 0 and i != 0:
                    self.val_step(i)
                    now_time = time.time()
                i += 1
                if i >= self.iter:
                    break
        self.writer.close()
        print("train end")

    def val_step(self, now_iter):
        self.model.eval()
        losses = 0
        i = 0
        with torch.no_grad():
            for _, (image, label, index) in enumerate(self.val_data_loader):
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.model(image)
                loss = self.seg_loss(output, label)
                losses += loss
                self.compute.update(label, output.argmax(1))
                i += 1
        acc_global, acc, iu, miou = self.compute.compute()
        mean_acc = sum(acc) / len(acc)
        losses = losses / i
        self.writer.add_scalar("val_loss", losses, now_iter)
        self.writer.add_scalar("val_miou", miou, now_iter)
        self.writer.add_scalar("mean_acc", mean_acc, now_iter)
        self.print("iu: {}".format(iu))
        self.print(
            "val_loss: {}, val_miou: {}, mean_acc: {}".format(
                losses, miou, mean_acc))
        if miou > self.pre_miou:
            self.pre_miou = miou
            model_name = "miou{}iter{}.pth".format(miou, now_iter)
            save_path = os.path.join(self.log_path, model_name)
            torch.save(self.model.state_dict(), save_path)
        self.compute.reset()
        self.model.train()

    def print(self, *arg):
        print(*arg)
        log_name = 'log.txt'
        filename = os.path.join(self.log_path, log_name)
        print(*arg, file=open(filename, "a"))
