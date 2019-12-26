"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR

from lib.networks import NetG,NetD, weights_init,Class
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.image_argu import rotate_img,rotate_img_trans
# from lib.mmd import mix_rbf_mmd2
from lib.getData import cifa10Data
from torch.utils.data import DataLoader
# from lib.mmd import huber_loss
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import random
from lib.evaluate import evaluate
import pickle

##


class SSnovelty(object):

    @staticmethod
    def name():
        """Return name of the class.
        """
        return 'SSnovelty'

    def __init__(self, opt):
        super(SSnovelty, self).__init__()
        ##
        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        # self.warmup = hyperparameters['model_specifics']['warmup']
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

        # -- Discriminator attributes.
        self.out_d_real = None
        self.feat_real = None
        self.err_d_real = None
        self.fake = None
        self.latent_i = None
        # self.latent_o = None
        self.out_d_fake = None
        self.feat_fake = None
        self.err_d_fake = None
        self.err_d = None
        self.idx = 0
        self.opt.display = True

        # -- Generator attributes.
        self.out_g = None
        self.err_g_bce = None
        self.err_g_l1l = None
        self.err_g_enc = None
        self.err_g = None

        # -- Misc attributes
        self.epoch = 0
        self.epoch1 = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        self.netc = Class(self.opt).to(self.device)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            # self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            # self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            # self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            self.netc.load_state_dict(torch.load(os.path.join(self.opt.resume, 'class.pth'))['state_dict'])
            print("\tDone.\n")

        # print(self.netg)
        # print(self.netd)

        ##
        # Loss Functions
        self.bce_criterion = nn.BCELoss()
        self.l1l_criterion = nn.L1Loss()
        self.mse_criterion = nn.MSELoss()
        self.l2l_criterion = l2_loss
        self.loss_func = torch.nn.CrossEntropyLoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.input_1 = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.label_r = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(self.opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label = 0

        base = 1.0
        sigma_list = [1, 2, 4, 8, 16]
        self.sigma_list = [sigma / base for sigma in sigma_list]

        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.netc.train()

            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_c = optim.Adam(self.netc.parameters(), lr=self.opt.lr_c, betas=(self.opt.beta1, 0.999))

    def set_input(self, input):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        self.input.data.resize_(input[0].size()).copy_(input[0])
        self.gt.data.resize_(input[1].size()).copy_(input[1])

        # Copy the first batch as the fixed input.
        if self.total_steps == self.opt.batchsize:
            self.fixed_input.data.resize_(input[0].size()).copy_(input[0])

    ##

    def update_netd(self):
        """
        Update D network: Ladv = |f(real) - f(fake)|_2
        """
        ##
        # Feature Matching.
        self.netd.zero_grad()
        # --
        # Train with real
        self.label.data.resize_(self.input.size(0)).fill_(self.real_label)
        self.out_d_real, self.feat_real = self.netd(self.input)
        # self.err_d_real = self.bce_criterion(self.out_d_real,self.label)

        # Train with fake
        self.label.data.resize_(self.input.size(0)).fill_(self.fake_label)
        self.fake, self.latent_i, = self.netg(self.input)
        self.out_d_fake, self.feat_fake = self.netd(self.fake.detach())
        # self.err_d_fake = self.bce_criterion(self.out_d_fake, self.label)

        # --
        # self.err_d = self.err_d_real + self.err_d_fake
        self.err_d = l2_loss(self.feat_real, self.feat_fake)
        # self.err_d = self.err_d_fake + self.err_d_l2

        # self.err_d_real = self.err_d
        # self.err_d_fake = self.err_d
        self.err_d.backward()

        self.optimizer_d.step()


    ##


    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('Reloading d net')


    def updata_netc(self):

        self.netc.zero_grad()

        output_real = self.netc(self.img_real)

        self.fake = self.netg(self.img_real)

        output_fake = self.netc(self.fake)


        self.err_c_fake = self.loss_func(output_fake, self.label_real)
        self.err_c_real = self.loss_func(output_real, self.label_real)
        self.err_c_dis = self.l2l_criterion(output_real,output_fake)
        self.err_c = self.err_c_fake + self.err_c_real + self.err_c_dis

        self.err_c.backward()
        self.optimizer_c.step()

    ##

    def trans_img(self,input):
        size = len(input)
        trans_map = torch.empty(size=(size, self.opt.nc, self.opt.isize, self.opt.isize),
                                    dtype=torch.float32,
                                    device=self.device)
        idx = int(size / 4)
        for i in range(idx):
            img = rotate_img_trans(input[i * 4 + 2],1)

            trans_map[i * 4 ] = img

            img = rotate_img_trans(input[i * 4 + 3],1)
            trans_map[i * 4 + 1] = img

            img = rotate_img_trans(input[i * 4 ],1)
            trans_map[i * 4 + 2] = img

            img = rotate_img_trans(input[i * 4 + 1],1)
            trans_map[i * 4 + 3] = img



        return trans_map


    def update_netg(self):
        """
        # ============================================================ #
        # (2) Update G network: log(D(G(x)))  + ||G(x) - x||           #
        # ============================================================ #

        """
        self.netg.zero_grad()


        # self.out_g, _ = self.netd(self.fake)
        # self.label.data.resize_(self.out_g.shape).fill_(self.real_label)
        # self.err_g_bce = self.bce_criterion(self.out_g, self.label)
        self.fake = self.netg(self.img_real)
        self.img_trans = self.trans_img(self.fake.detach().cpu())
        self.err_g_r = self.mse_criterion(self.fake, self.img_trans)
        # self.err_g_l1l = self.mse_criterion(self.fake, self.img_real)  # constrain x' to look like x
        # self.err_g_enc = self.l2l_criterion(self.latent_o, self.latent_i)
        output_fake = self.netc(self.fake)
        output_real = self.netc(self.img_real)
        self.err_g_loss = self.l2l_criterion(output_fake, output_real)
        self.loss = self.loss_func(output_fake, self.label_real)

        # self.err_g = self.err_g_bce + self.err_g_l1l * self.opt.w_rec + (self.loss + self.err_g_loss) * self.opt.w_enc
        # self.err_g = self.err_g_bce + (loss + self.err_g_loss) * self.opt.w_enc

        # self.err_g = (self.loss + self.err_g_loss ) * self.opt.w_enc

        self.err_g = (self.err_g_r ) * self.opt.w_rec + (self.loss + self.err_g_loss) * self.opt.w_enc
        self.err_g.backward(retain_graph=True)
        self.optimizer_g.step()

    ##

    def argument_image_rotation_plus(self, X):
        size = len(X)
        self.img_real = torch.empty(size=(size * 4, self.opt.nc, self.opt.isize, self.opt.isize),
                                    dtype=torch.float32,
                                    device=self.device)
        self.label_real = torch.empty(size=(size * 4,), dtype=torch.long, device=self.device)
        for idx in range(size):
            img0 = X[idx]
            for i in range(4):
                [img, label] = rotate_img(img0, i)
                self.img_real[idx * 4 + i] = img
                self.label_real[idx * 4 + i] = label

    def optimize(self):
        """ Optimize netD and netG  networks.
        """

        self.argument_image_rotation_plus(self.input)
        self.input_img = self.input.to(self.device)



        self.updata_netc()
        # self.update_netd()
        self.update_netg()

        # If D loss is zero, then re-initialize netD
        # if self.err_d_real.item() < 1e-5 or self.err_d_fake.item() < 1e-5:
        #     self.reinitialize_netd()

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([('err_d', self.err_d.item()),
                              ('err_g', self.err_g.item()),
                              ])

        return errors

    def get_errors_1(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        errors = OrderedDict([('err_c_real', self.err_c_real.item()),
                              ('err_c_fake', self.err_c_fake.item()),
                              ('err_c', self.err_c.item()),])

        return errors
    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.img_real.data
        fakes = self.fake.data
        trans = self.img_trans.data
        # fixed = self.netg(self.fixed_input)[0].data
        # fixed_input = self.fixed_input.data

        return reals, fakes, trans

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))

    ##


    def train_step(self):
        self.netg.train()
        epoch_iter = 0
        for step, (x, y, z) in enumerate(self.train_loader):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize
            self.input = Variable(x)


            self.optimize()

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, trans  = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, trans)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, trans)

            # errors = self.get_errors()
            # if self.total_steps % self.opt.save_image_freq == 0:
            #     reals, fakes, fixed , fixed_input = self.get_current_images()
            #     self.visualizer.save_current_images(self.epoch, reals, fakes, fixed )
            #     if self.opt.display:
            #         self.visualizer.display_current_images(reals, fakes, fixed,fixed_input)

        # print('Epoch %d  err_g %f err_d %f  err_c %f' % (self.epoch, self.err_g.item(),self.err_d.item(),self.err_c.item()))
        print('Epoch %d  err_g %f err_c %f' % (self.epoch, self.err_g.item(), self.err_c.item()))
        # print('Epoch %d  err_d_real %f err_d_fake %f ' % (self.epoch, self.err_d_real.item(), self.err_d_fake.item()))
        # print('Epoch %d  err_g_bce %f err_g_loss %f err_g_l1 %f loss %f ' % (self.epoch, self.err_g_bce.item(),
        #                                                   self.err_g_loss.item(),self.err_g_l1l.item(),self.loss.item()))



        # print(">> Training model %s. Epoch %d/%d" % (self.name(), self.epoch + 1, self.opt.niter))


    def train(self):
        """ Train the model
        """
        ##
        # TRAIN
        self.total_steps = 0
        best_auc = [0, 0, 0]


        # Train for niter epochs.
        print(">> Training model %s." % self.name())
        train_data, test_data = cifa10Data(self.opt.normalclass)
        self.train_loader = DataLoader(train_data, batch_size=self.opt.batchsize, shuffle=True, num_workers=0,pin_memory=True)
        self.test_loader = DataLoader(test_data, batch_size=self.opt.batchsize, shuffle=False, num_workers=0,pin_memory=True)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_step()
            if self.epoch%20 == 0:
                rec  = self.test()
                if rec['AUC_R'] > best_auc[0]:
                    best_auc[0] = rec['AUC_R']
                if rec['AUC_C_real'] > best_auc[1]:
                    best_auc[1] = rec['AUC_C_real']
                if rec['AUC_C_fake'] > best_auc[2]:
                    best_auc[2] = rec['AUC_C_fake']

                # self.visualizer.print_current_performance(rec, best_auc)

                f = open('./output/testclass.txt', 'a', encoding='utf-8-sig', newline='\n')
                #

                f.write('rec: '  + str(rec['AUC_R'], ) + ', ' + str(rec['AUC_C_real'], ) + ',' + str(rec['AUC_C_fake'], ) + '\n')
                f.write('best' + str(best_auc) + '\n')
                f.close()
                # self.test_1()
            # self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name())
        self.test()
        # self.test_1()


    ##

    def test(self):
        with torch.no_grad():

            self.opt.load_weights = True
            self.epoch1 = 1
            self.epoch2 = 200

            self.total_steps = 0
            epoch_iter = 0
            print('test')
            label = torch.zeros(size=(10000,), dtype=torch.long, device=self.device)
            pre = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)
            pre_real = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)

            self.relation = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)
            self.distance = torch.zeros(size=(40000,), dtype=torch.float32, device=self.device)
            self.relation_img = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)
            self.distance_img = torch.zeros(size=(40000,), dtype=torch.float32, device=self.device)
            self.opt.phase = 'test'


            for i, (x, y, z) in enumerate(self.test_loader):
                self.input = Variable(x)
                self.label_r = Variable(z)
                self.total_steps += self.opt.batchsize

                self.argument_image_rotation_plus(self.input)
                self.label_r = self.label_r.to(self.device)

                classfiear_real_1 = self.netc(self.img_real)
                classfiear_real = F.softmax(classfiear_real_1, dim=1)

                prediction_real = -(torch.log(classfiear_real))
                self.fake = self.netg(self.img_real)
                classfiear_1 = self.netc(self.fake)

                classfiear = F.softmax(classfiear_1, dim=1)

                prediction = -(torch.log(classfiear))
                aaaa = (prediction.size(0) / 4)
                aaaa = int(aaaa)
                # prediction = prediction * (-1/4)

                label_z = torch.zeros(size=(aaaa,), dtype=torch.long, device=self.device)
                pre_score = torch.zeros(size=(aaaa,), dtype=prediction.dtype, device=self.device)
                pre_score_real = torch.zeros(size=(aaaa,), dtype=prediction.dtype, device=self.device)

                self.img_trans = self.trans_img(self.fake.cpu())

                distance_img = torch.mean(torch.pow((self.fake - self.img_real), 2), -1)
                distance_img = torch.mean(torch.mean(distance_img, -1), -1)

                if self.total_steps % self.opt.save_image_freq == 0:
                    reals, fakes, trans = self.get_current_images()
                    self.visualizer.save_test_images(i, reals, fakes, trans)
                    if self.opt.display:
                        self.visualizer.display_test_images(reals, fakes, trans)

                # distance = torch.mean(torch.pow((classfiear_1 - classfiear_real_1), 2), -1)

                # self.distance[i * self.opt.batchsize: i * self.opt.batchsize + distance.size(0)] = distance.reshape(
                #     distance.size(0))

                self.distance_img[i * 64: i * 64 + distance_img.size(0)] = distance_img.reshape(
                    distance_img.size(0))

                for k in range(aaaa):
                    # label_z[k] = self.label_r[k * 4]
                    pre_score[k] = (prediction[k * 4, 0] + prediction[k * 4 + 1, 1] +
                                    prediction[k * 4 + 2, 2] + prediction[k * 4 + 3, 3]) / 4
                    pre_score_real[k] = (prediction_real[k * 4, 0] + prediction_real[k * 4 + 1, 1] +
                                         prediction_real[k * 4 + 2, 2] + prediction_real[k * 4 + 3, 3]) / 4

                label[i * self.opt.batchsize: i * self.opt.batchsize + aaaa] = self.label_r
                pre[i * self.opt.batchsize: i * self.opt.batchsize + aaaa] = pre_score
                pre_real[i * self.opt.batchsize: i * self.opt.batchsize + aaaa] = pre_score_real

            for j in range(10000):
                # self.relation[j] = self.distance[j*4]
                self.relation_img[j] = self.distance_img[j * 4]



            # D = pre + self.relation * 0.2
            # D_real = pre_real + self.relation * 0.2
            #
            # mu = torch.mul(pre, self.relation)
            # mu_real = torch.mul(pre_real, self.relation)
            aaaa = self.relation_img.cpu().numpy()
            np.savetxt('./output/log.txt', aaaa)
            bbbb = label.cpu().numpy()
            np.savetxt('./output/label.txt', bbbb)

            # auc_mu_fake = evaluate(label, mu, metric=self.opt.metric)
            # auc_mu_real = evaluate(label, mu_real, metric=self.opt.metric)
            # auc_d_fake = evaluate(label, D, metric=self.opt.metric)
            # auc_d_real = evaluate(label, D_real, metric=self.opt.metric)
            auc_c_fake = evaluate(label, pre, metric=self.opt.metric)
            auc_c_real = evaluate(label, pre_real, metric=self.opt.metric)
            # auc_r = evaluate(label, self.relation, metric=self.opt.metric)
            auc_r_img = evaluate(label, self.relation_img, metric=self.opt.metric)

            performance = OrderedDict([('AUC_R', auc_r_img),
                                       ('AUC_C_real', auc_c_real),
                                       ('AUC_C_fake', auc_c_fake)])
            print('test done')
            return performance

            # print('Train mul_real ROC AUC Score: %f  mu_fake: %f' % (auc_mu_real, auc_mu_fake))
            # print('Train add_real ROC AUC Score: %f  add_fake: %f' % (auc_d_real, auc_d_fake))






    def test_1(self):

        with torch.no_grad():

            self.total_steps_test = 0
            epoch_iter = 0
            print('test')
            label = torch.zeros(size=(10000,), dtype=torch.long, device=self.device)
            pre = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)
            pre_real = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)

            self.relation = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)
            self.relation_img = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)

            self.classifiear = torch.zeros(size=(10000,4), dtype=torch.float32, device=self.device)
            self.opt.phase = 'test'
            for i, (x, y, z) in enumerate(self.test_loader):
                self.input = Variable(x)
                self.label_rrr = Variable(z)

                self.input = self.input.to(self.device)
                self.label_rrr = self.label_rrr.to(self.device)


                size = int(self.input.size(0)/4)
                input_1 = torch.empty(size=(size, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                      device=self.device)
                input_2 = torch.empty(size=(size, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                      device=self.device)
                input_3 = torch.empty(size=(size, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                      device=self.device)
                input_4 = torch.empty(size=(size, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                      device=self.device)


                classfiear_real_1 = self.netc(self.input)
                classfiear_real = F.softmax(classfiear_real_1, dim=1)

                prediction_real = -(torch.log(classfiear_real))
                for j in range(size):
                    input_1[j] = self.input[j*4]
                    input_2[j] = self.input[j * 4 +1]
                    input_3[j] = self.input[j * 4 +2]
                    input_4[j] = self.input[j * 4 +3]
                output_1 = self.netg(input_1)
                output_2 = self.netg(input_2)
                output_3 = self.netg(input_3)
                output_4 = self.netg(input_4)
                classifiear_real = self.netc(input_1)

                classfiear_11 = self.netc(output_1)
                classfiear_21 = self.netc(output_2)
                classfiear_31 = self.netc(output_3)
                classfiear_41 = self.netc(output_4)




                classfiear_1 = F.softmax(classfiear_11, dim=1)
                classfiear_2 = F.softmax(classfiear_21, dim=1)
                classfiear_3 = F.softmax(classfiear_31, dim=1)
                classfiear_4 = F.softmax(classfiear_41, dim=1)

                prediction_1 = -(torch.log(classfiear_1))
                prediction_2 = -(torch.log(classfiear_2))
                prediction_3 = -(torch.log(classfiear_3))
                prediction_4 = -(torch.log(classfiear_4))

                aaaa = prediction_1.size(0)
                self.classifiear[i * 16: i * 16 + aaaa] = classfiear_11

                # prediction = prediction * (-1/4)

                label_z = torch.zeros(size=(aaaa,), dtype=torch.long, device=self.device)
                pre_score = torch.zeros(size=(aaaa,), dtype=prediction_1.dtype, device=self.device)
                pre_score_real = torch.zeros(size=(aaaa,), dtype=prediction_1.dtype, device=self.device)

                distance_img = torch.mean(torch.pow((output_1 - input_1), 2), -1)
                distance_img = torch.mean(torch.mean(distance_img, -1), -1)

                distance = torch.mean(torch.pow((classifiear_real - classfiear_11), 2), -1)


                self.relation[i * 16: i * 16 + distance.size(0)] = distance.reshape(distance.size(0))
                self.relation_img[i * 16: i * 16 + distance.size(0)] = distance_img.reshape(distance.size(0))

                for k in range(aaaa):
                    label_z[k] = self.label_rrr[k * 4]
                    pre_score[k] = (prediction_1[k , 0] + prediction_2[k , 1] +
                                    prediction_3[k , 2] + prediction_4[k , 3]) / 4
                    pre_score_real[k] = (prediction_real[k * 4, 0] + prediction_real[k * 4 + 1, 1] +
                                         prediction_real[k * 4 + 2, 2] + prediction_real[k * 4 + 3, 3]) / 4

                label[i * 16: i * 16 + aaaa] = label_z
                pre[i * 16: i * 16 + aaaa] = pre_score
                pre_real[i * 16: i * 16 + aaaa] = pre_score_real

            D = pre + self.relation * 0.2
            D_real = pre_real + self.relation * 0.2

            aaaa = self.classifiear.cpu().numpy()
            np.savetxt('./output/log.txt', aaaa)
            bbbb = label.cpu().numpy()
            np.savetxt('./output/label.txt', bbbb)

            mu = torch.mul(pre, self.relation)
            mu_real = torch.mul(pre_real, self.relation)

            auc_mu_fake = evaluate(label, mu, metric=self.opt.metric)
            auc_mu_real = evaluate(label, mu_real, metric=self.opt.metric)
            auc_d_fake = evaluate(label, D, metric=self.opt.metric)
            auc_d_real = evaluate(label, D_real, metric=self.opt.metric)
            auc_c_fake = evaluate(label, pre, metric=self.opt.metric)
            auc_c_real = evaluate(label, pre_real, metric=self.opt.metric)
            auc_r = evaluate(label, self.relation, metric=self.opt.metric)
            auc_r_img = evaluate(label, self.relation_img, metric=self.opt.metric)

            print('Train mul_real ROC AUC Score: %f  mu_fake: %f' % (auc_mu_real, auc_mu_fake))
            print('Train add_real ROC AUC Score: %f  add_fake: %f' % (auc_d_real, auc_d_fake))
            print('Train class_real ROC AUC Score: %f class_fake: %f' % (auc_c_real, auc_c_fake))

            print('Train recon ROC AUC Score: %f recon_img:%f' % (auc_r,auc_r_img))
            print('test done')















