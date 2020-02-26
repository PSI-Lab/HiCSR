import abc
import argparse
import os
import logging.config
import yaml
import json
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.HiCSR_model import Generator, Discriminator
from models.DAE_model import DAE
from feature_reconstruction_loss import FeatureReconstructionLoss

from dataloader import DatasetFromFolder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_logging(log_dir, default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    '''
    Setup logging configuration, config file is logging.yaml
    '''
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        config['handlers']['debug_file_handler']['filename'] = log_dir
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

class ClassModel(object):
    __metaclass__ = abc.ABCMeta

    _hparams = {}
    _model = None
    _input_shape = None
    _output_shape = None
    _writer = None

    def set_data_shapes(self, input_shape, output_shape):
        self._input_shape = input_shape
        self._output_shape = output_shape

    def set_writer(self, summary_writer):
        self._writer = summary_writer

    @abc.abstractmethod
    def set_model(self, hparams):
        pass

    @abc.abstractmethod
    def fit_model(self, train_set):
        pass

    @abc.abstractmethod
    def _set_default_model_specific_hparams(self):
        pass

    def set_hparams(self, hparams_args={}):
        hparams = {
                "random_state" : None,
                }
        self._hparams = hparams
        self._set_default_model_specific_hparams()
        self._hparams.update(hparams_args)
        self._hparams.update(hparams)

    def update_hparams(self, hparams_args):
        self._hparams.update(hparams_args)

    def save_model(self):
        model_name = self.__class__.__name__
        logging.info("saving model as {}.pkl...".format(model_name))
        self._writer = None
        torch.save(self._model.state_dict(),
                   './experiments/{}/{}.pth'.format(self._hparams['experiment'],
                       self._hparams['experiment']))

        with open('./experiments/{}/{}.pkl'.format(self._hparams['experiment'],model_name), 'wb') as output:
            pickle.dump(self.__dict__, output, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("save complete")

    def load_model(self, filename):
        logging.info("loading model...")
        with open(filename+'.pkl', 'rb') as model_file:
            self.__dict__.clear()
            self.__dict__.update(pickle.load(model_file))
            self._model = torch.load(filename+'.pth')
        logging.info("model loaded")

class HiCSRModel(ClassModel):

    def _set_default_model_specific_hparams(self):
            self._hparams.update({
                "lambda_a": 2.5e-3,
                "lambda_f": 1,
                "lambda_1": 1,
                "G_optimizer": 'adam',
                "G_learning_rate": 1e-5,
                "D_optimizer": 'adam',
                "D_learning_rate": 1e-5,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "res_blocks": 15,
                "epochs": 500,
                "num_workers":8,
                "batch_size":128,
                "random_state": 12345,
                })

    def set_model(self):
        device = torch.device("cuda:{}".format(self._hparams['gpu']) if torch.cuda.is_available() else "cpu")
        logging.info('setting model on device: {}'.format(device))

        if torch.cuda.device_count() > 1:
            logging.info('Using {} devices'.format(torch.cuda.device_count()))
            G = Generator(num_res_blocks=self._hparams['res_blocks'])
            D = Discriminator()
            G.init_params()
            D.init_params()
            G = nn.DataParallel(G).cuda()
            D = nn.DataParallel(D).cuda()
        else:
            G = Generator(num_res_blocks=self._hparams['res_blocks']).to(device)
            D = Discriminator().to(device)
            G.init_params()
            D.init_params()

        self._model = G
        self.D = D

    def fit_model(self, train_set, valid_set):
        device = torch.device("cuda:{}".format(self._hparams['gpu']) if torch.cuda.is_available() else "cpu")
        logging.info('fitting model on device: {}'.format(device))

        train_loader = DataLoader(dataset=train_set,
                                  num_workers=self._hparams['num_workers'],
                                  batch_size=self._hparams['batch_size'],
                                  shuffle=True)

        valid_loader = DataLoader(dataset=valid_set,
                                  num_workers=self._hparams['num_workers'],
                                  batch_size=self._hparams['batch_size'],
                                  shuffle=False)

        G_optimizer = Adam(self._model.parameters(),lr=self._hparams['G_learning_rate'])

        D_optimizer = Adam(self.D.parameters(),lr=self._hparams['D_learning_rate'])

        adv_loss = nn.BCEWithLogitsLoss().to(device)
        l1_loss = nn.L1Loss().to(device)
        feature_reconstruction_loss = FeatureReconstructionLoss().to(device)

        logging.info("beginning adversarial training")
        for epoch in range(0, self._hparams['epochs']):
            train_G_total_epoch_loss = 0
            train_G_adv_epoch_loss = 0
            train_G_image_epoch_loss = 0
            train_G_feature_epoch_loss = 0

            D_epoch_loss = 0
            D_real_epoch_loss = 0
            D_fake_epoch_loss = 0
            D_real_epoch_acc = 0
            D_fake_epoch_acc = 0

            self._model.train()
            self.D.train()

            for iteration, (target, data) in enumerate(train_loader):

                #######################
                #       Train G       #
                #######################

                self._model.zero_grad()
                target, data = target.float().to(device), data.float().to(device)

                # I_sr = G(lr)
                output = self._model(data)

                # compute pixelwise loss
                image_loss = l1_loss(output, target)

                # compute feature reconstruction loss
                feature_loss = sum(feature_reconstruction_loss(output, target))

                # compute adversarial loss
                pred_fake = self.D(output)
                labels_real = torch.ones_like(pred_fake, requires_grad=False).to(device)
                GAN_loss = adv_loss(pred_fake, labels_real)

                # compute total generator loss
                total_loss_G = self._hparams['lambda_a'] * GAN_loss + \
                               self._hparams['lambda_1'] * image_loss + \
                               self._hparams['lambda_f'] * feature_loss

                # gradient step for generator
                total_loss_G.backward()
                G_optimizer.step()

                # record losses
                train_G_total_epoch_loss += total_loss_G.item()
                train_G_adv_epoch_loss += GAN_loss.item()
                train_G_image_epoch_loss += image_loss.item()
                train_G_feature_epoch_loss += feature_loss

                #######################
                #       Train D       #
                #######################

                self.D.zero_grad()

                # train on real data
                pred_real = self.D(target)
                labels_real = torch.ones_like(pred_real, requires_grad=False).to(device)
                pred_labels_real = (pred_real>0.5).float().detach()
                acc_real = (pred_labels_real == labels_real).float().sum()/labels_real.shape[0]
                loss_real = adv_loss(pred_real, labels_real)
                loss_real.backward()

                # train on fake data
                output = self._model(data)
                pred_fake = self.D(output.detach())
                labels_fake = torch.zeros_like(pred_fake, requires_grad=False).to(device)
                pred_labels_fake = (pred_fake>0.5).float().detach()
                acc_fake = (pred_labels_fake == labels_fake).float().sum()/labels_fake.shape[0]
                loss_fake = adv_loss(pred_fake, labels_fake)
                loss_fake.backward()

                # get total loss
                total_loss_D = loss_real + loss_fake

                # gradient step for discriminator
                D_optimizer.step()

                # record losses
                D_epoch_loss += total_loss_D.item()
                D_real_epoch_loss += loss_real.item()
                D_fake_epoch_loss += loss_fake.item()
                D_real_epoch_acc += acc_real.item()
                D_fake_epoch_acc += acc_fake.item()

                # log training progress
                if iteration%10 == 0:
                    logging.info("===> Training Epoch[{}]({}/{}) "
                                 "[G: {:.4f} l1 loss: {:.4f} adv loss: {:.4f} feat loss: {:.4f}] "
                                 "[D: {:.4f} real_loss: {:.4f} real_acc: {:.4f} "
                                 "fake_loss: {:.4f} fake_acc: {:.4f}]".format(epoch,
                                                                              iteration,
                                                                              len(train_loader),
                                                                              total_loss_G.item(),
                                                                              image_loss.item(),
                                                                              GAN_loss.item(),
                                                                              feature_loss,
                                                                              total_loss_D.item(),
                                                                              loss_real.item(),
                                                                              acc_real.item(),
                                                                              loss_fake.item(),
                                                                              acc_fake.item()))
            with torch.no_grad():
                valid_epoch_image_loss = 0
                self._model.eval()
                for i, (target, data) in enumerate(valid_loader):
                    target, data = target.float().to(device), data.float().to(device)
                    output = self._model(data).detach()
                    image_loss = l1_loss(target, output)
                    valid_epoch_image_loss += image_loss.item()

            #######################
            #       logging       #
            #######################

            valid_image_loss = valid_epoch_image_loss/len(valid_loader)
            D_real_loss = D_real_epoch_loss/len(train_loader)
            D_real_acc = D_real_epoch_acc/len(train_loader)
            D_fake_loss = D_fake_epoch_loss/len(train_loader)
            D_fake_acc = D_fake_epoch_acc/len(train_loader)
            D_loss = D_epoch_loss/len(train_loader)
            train_G_total_loss = train_G_total_epoch_loss/len(train_loader)
            train_G_image_loss = train_G_image_epoch_loss/len(train_loader)
            train_G_adv_loss = train_G_adv_epoch_loss/len(train_loader)
            train_G_feature_loss = train_G_feature_epoch_loss/len(train_loader)


            self._log_epoch_losses(self._writer,
                                   epoch,
                                   D_loss,
                                   D_real_loss,
                                   D_real_acc,
                                   D_fake_loss,
                                   D_fake_acc,
                                   train_G_total_loss,
                                   train_G_image_loss,
                                   train_G_adv_loss,
                                   train_G_feature_loss,
                                   valid_image_loss)

            self._log_epoch_images(self._writer, epoch, data, target, output, 20)

            if epoch%20 == 0:
                self._save_checkpoint(epoch, G_optimizer, D_optimizer)

    def _log_epoch_losses(self,
                          summary_writer,
                          epoch,
                          D_loss,
                          D_loss_real,
                          D_acc_real,
                          D_loss_fake,
                          D_acc_fake,
                          train_G_total_loss,
                          train_G_image_loss,
                          train_G_adv_loss,
                          train_G_feature_loss,
                          valid_image_loss):
        summary_writer.add_scalar('D_loss', D_loss, epoch)
        summary_writer.add_scalars('D_loss_components', {'real': D_loss_real, 'fake':D_loss_fake}, epoch)
        summary_writer.add_scalars('D_acc_components', {'real': D_acc_real, 'fake':D_acc_fake}, epoch)
        summary_writer.add_scalars('G_loss', {'train': train_G_total_loss, 'valid': valid_image_loss}, epoch)
        summary_writer.add_scalar('image_loss', train_G_image_loss, epoch)
        summary_writer.add_scalar('adversarial_loss', train_G_adv_loss, epoch)
        summary_writer.add_scalar('feature_loss', train_G_feature_loss, epoch)
        summary_writer.flush()

        logging.info("Epoch {} Complete: \n[D Loss: {:.4f} Real D Loss: {:.4f} Fake D_loss: {:.4f}] [Train G Loss: {:.4f} Valid G loss: {:.4f}]\n".format(epoch,
                                                                                                                                                          D_loss, D_loss_real, D_loss_fake,
                                                                                                                                                          train_G_total_loss, valid_image_loss))
    def _log_epoch_images(self, summary_writer, epoch, data, target, output, n_images):
        data = data.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        output = output.cpu().detach().numpy()

        for i in range(0,n_images):
            fig, axs = plt.subplots(1,3, facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace = .5, wspace=.1)
            axs = axs.ravel()
            for j, mat in enumerate([data[i],target[i],output[i]]):
                if mat.shape[-1] == 40:
                    mat = mat[:,6:34,6:34]
                im = axs[j].matshow(mat[0], cmap='YlOrRd', interpolation="none", vmin=-1, vmax=1)
                plt.setp(axs[j].get_xticklabels(), visible=False)
                plt.setp(axs[j].get_yticklabels(), visible=False)
                axs[j].tick_params(axis='both', which='both', length=0)
            plt.title('input/target/prediction')
            summary_writer.add_figure('epoch_image_comparison_{}'.format(i),fig,epoch)
            summary_writer.flush()

    def _save_checkpoint(self, epoch, G_optimizer, D_optimizer, scheduler = None):
        state = {
                'epoch': epoch,
                'G_state_dict': self._model.state_dict(),
                'D_state_dict': self.D.state_dict(),
                'G_optimizer': G_optimizer.state_dict(),
                'D_optimizer': D_optimizer.state_dict(),
                'scheduler': scheduler,
                }
        torch.save(state, './experiments/{}/checkpoints/{}_ckpt_{}.pth'.format(self._hparams['experiment'], self.__class__.__name__, epoch))

class DAEModel(ClassModel):
    def _set_default_model_specific_hparams(self):
            self._hparams.update({
                "batch_size": 256,
                "epochs": 600,
                "learning_rate": 0.0001,
                "noise_scale": 0.1,
                "gpu": 0,
                "num_workers": 8,
                "random_state": 12345,
                })

    def set_model(self):
        torch.manual_seed(self._hparams['random_state'])
        device = torch.device("cuda:{}".format(self._hparams['gpu']) if torch.cuda.is_available() else "cpu")
        logging.info('setting model on device: {}'.format(device))

        if torch.cuda.device_count() > 1:
            logging.info('Using {} devices'.format(torch.cuda.device_count()))
            net = nn.DataParallel(DAE().to(device))
        else:
            net = DAE().to(device)

        net = DAE().to(device)
        self._model = net

    def fit_model(self, train_set, valid_set):
        device = torch.device("cuda:{}".format(self._hparams['gpu']) if torch.cuda.is_available() else "cpu")
        logging.info('fitting model on device: {}'.format(device))

        train_loader = DataLoader(dataset=train_set,
                                  num_workers=self._hparams['num_workers'],
                                  batch_size=self._hparams['batch_size'],
                                  shuffle=True)

        valid_loader = DataLoader(dataset=valid_set,
                                  num_workers=self._hparams['num_workers'],
                                  batch_size=self._hparams['batch_size'],
                                  shuffle=False)

        optimizer = Adam(self._model.parameters(), lr=self._hparams['learning_rate'])

        criterion = nn.MSELoss()

        for epoch in range(0, self._hparams['epochs']):
            train_epoch_loss = 0
            for iteration, (target, _) in enumerate(train_loader):
                self._model.train()
                self._model.zero_grad()
                target = target.float().to(device)
                target_noisy = target + self._hparams['noise_scale'] * torch.randn_like(target)
                target_noisy = torch.clamp(target_noisy, min=-1, max=1)
                output = self._model(target_noisy)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()

                if iteration%10 == 0:
                    logging.info("===> Training Epoch[{}]({}/{}): Loss: {:.8f}".format(epoch, iteration, len(train_loader), loss.item()))

            valid_epoch_loss = 0
            with torch.no_grad():
                for iteration, (target, _) in enumerate(valid_loader):
                    self._model.eval()
                    target = target.float().to(device)
                    target_noisy = target + self._hparams['noise_scale'] * torch.randn_like(target)
                    target_noisy = torch.clamp(target_noisy, min=-1, max=1)
                    output = self._model(target)
                    loss = criterion(output, target)
                    valid_epoch_loss += loss.item()

            train_epoch_loss = train_epoch_loss/len(train_loader)
            valid_epoch_loss = valid_epoch_loss/len(valid_loader)
            self._log_epoch_losses(self._writer, epoch, train_epoch_loss, valid_epoch_loss)

            target = target.cpu().detach().numpy()
            target_noisy = target_noisy.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            self._log_epoch_images(self._writer, epoch, target_noisy, target, output, 30)

            if epoch%20 == 0:
                torch.save(self._model,'./experiments/{}/checkpoints/{}_ckpt_{}.pth'.format(self._hparams['experiment'], self.__class__.__name__, epoch))

    def _log_epoch_losses(self, summary_writer, epoch, train_loss, valid_loss):
        summary_writer.add_scalars('mse', {'validation loss':valid_loss, 'training loss':train_loss}, epoch)
        summary_writer.flush()

        logging.info("Epoch {} Complete: Avg. Train Loss: {:.8f} - Avg. Valid Loss: {:.8f}\n".format(epoch, train_loss, valid_loss))

    def _log_epoch_images(self, summary_writer, epoch, target_noisy, target, output, n_images):
        if epoch%5 == 0:
            for i in range(0,n_images):
                fig, axs = plt.subplots(1,3, facecolor='w', edgecolor='k')
                fig.subplots_adjust(hspace = .5, wspace=.1)
                axs = axs.ravel()
                for j, mat in enumerate([target_noisy[i],target[i],output[i]]):
                    if mat.shape[-1] == 40:
                        mat = mat[:,6:34,6:34]
                    axs[j].matshow(mat[0], cmap='YlOrRd', interpolation="none", vmin=-1, vmax=1)
                    plt.setp(axs[j].get_xticklabels(), visible=False)
                    plt.setp(axs[j].get_yticklabels(), visible=False)
                    axs[j].tick_params(axis='both', which='both', length=0)
                plt.title('noisy_input/target/prediction')
                summary_writer.add_figure('epoch_image_comparison_{}'.format(i),fig,epoch)
                summary_writer.flush()

MODEL_REGISTRY = {
            "HiCSR":HiCSRModel,
            "DAE":DAEModel,
            }

def main():
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--data_fp', type=str, required=True,
           			help="directory containing training and validation data to use for model training")
    model_parser.add_argument('--model', type=str, required=True,
            help="Set the model to be trained, There are two options: 'HiCSR' will train the Hi-C enhancement model and 'DAE' will train the Denoising Autoencoder on the high resolution data")
    model_parser.add_argument('--gpu', type=int, default=0,
				help="GPU number to use for training, if the system has no GPU, training will automatically default to using the CPU. default = 0.")
    model_parser.add_argument('--experiment', type=str, required=True,
                                help="experiment name associated with the training run, all model logging and final model file are saved under this name. Experiment name must match an entry in the experiment_hyperparameters.json config file")
    args = model_parser.parse_args()

    with open('experiment_hyperparameters.json', 'r') as f:
        experiment_queue = json.load(f)
        assert args.experiment in experiment_queue

    experiment_specific_hparams = experiment_queue[args.experiment]

    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    experiment_fp = './experiments/{}'.format(args.experiment)
    if not os.path.exists(experiment_fp):
        os.mkdir(experiment_fp)
        os.mkdir(experiment_fp+'/logs/')
        os.mkdir(experiment_fp+'/checkpoints/')
        os.mkdir(experiment_fp+'/tensorboard/')

    setup_logging(log_dir='./experiments/{}/logs/{}.log'.format(args.experiment, args.experiment))
    writer = SummaryWriter(log_dir='./experiments/{}/tensorboard/'.format(args.experiment),comment=args.experiment)

    train_set = DatasetFromFolder(args.data_fp, data_type='train')
    valid_set = DatasetFromFolder(args.data_fp, data_type='valid')

    model = MODEL_REGISTRY[args.model]()
    logging.info("defining model: {}".format(model.__class__.__name__))
    model.set_hparams(vars(args))
    model.update_hparams(experiment_specific_hparams)
    logging.info("setting tensorboard writer")
    model.set_writer(writer)
    logging.info("hyperparameters: {}".format(model._hparams))
    hr_shape, lr_shape = train_set.get_shape()
    logging.info("train/valid samples: {}/{}".format(len(train_set), len(valid_set)))
    logging.info("input/output shape: {}/{}".format(lr_shape, hr_shape))
    model.set_data_shapes(input_shape=lr_shape, output_shape=hr_shape)
    model.set_model()
    model.fit_model(train_set, valid_set)
    model.save_model()

if __name__ == "__main__":
    main()
