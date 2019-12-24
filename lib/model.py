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
from lib.mmd import mix_rbf_mmd2
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR

from lib.networks import NetG,NetD, weights_init,Class
from lib.visualizer import Visualizer
from lib.loss import l2_loss,attention_loss
# from lib.mmd import mix_rbf_mmd2
from lib.getData import cifa10Data
from torch.utils.data import DataLoader
# from lib.mmd import huber_loss
import torch.nn.functional as F
from lib.image_argu import rotate_img,rotate_img_recon
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import random
from lib.evaluate import evaluate
import pickle

import numpy

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

        self.img_real = torch.empty(size=(self.opt.batchsize * 4, self.opt.nc, self.opt.isize, self.opt.isize),
                                    dtype=torch.float32,
                                    device=self.device)
        self.img_fake = torch.empty(size=(self.opt.batchsize * 4, self.opt.nc, self.opt.isize, self.opt.isize),
                                    dtype=torch.float32,
                                    device=self.device)
        self.label_fake = torch.empty(size=(self.opt.batchsize * 4,), dtype=torch.long, device=self.device)

        self.label_real = torch.empty(size=(self.opt.batchsize * 4,), dtype=torch.long, device=self.device)

        self.output = torch.empty(size=(self.opt.batchsize * 4, 4),dtype=torch.float32,device=self.device)


        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

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
            self.optimizer_dis = optim.Adam(self.netc.parameters(), lr=self.opt.lr_c, betas=(self.opt.beta1, 0.999))

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



    def max_act(self,input):
        loc_label = torch.ones([input.size(0), 2])
        loc_label = loc_label.cuda()
        len  =input.size(0)


        for j in range(len):
            response_map = input[j]

            response_map = torch.unsqueeze(response_map,0)
            response_map = F.upsample(response_map, size=[32, 32])
            response_map = torch.squeeze(response_map, 0)

            #
            response_map = response_map.mean(0)
            rawmaxidx = response_map.view(-1).max(0)[1]
            idx = []
            for d in list(response_map.size())[::-1]:
                idx.append(rawmaxidx % d)
                rawmaxidx = rawmaxidx / d
            loc_label[j, 0] = (idx[1].float() + 0.2)
            loc_label[j, 1] = (idx[0].float() + 0.2)
        return loc_label


    def act_label(self,input):
        idx = int(len(input) / 4)
        labels = torch.empty(size=(input.size(0),2), dtype=torch.float32, device=self.device)

        for i in range(idx):
            x0, y0 = input[i * 4][0], input[i * 4][1]
            x1, y1 = input[i * 4 + 1][0], input[i * 4 + 1][1]
            x2, y2 = input[i * 4 + 2][0], input[i * 4 + 2][1]
            x3, y3 = input[i * 4 + 2][0], input[i * 4 + 2][1]
            labels[i * 4] = y3
            labels[i * 4, 1] = 32-x3
            labels[i * 4 + 1,0] = 32-y0
            labels[i * 4 + 1,1] = x0
            labels[i * 4 + 2,0] = 32-y1
            labels[i * 4 + 2,1] = x1
            labels[i * 4 + 3,0] = y2
            labels[i * 4 + 2,1] = 32-x2

        return labels

    def updata_netc(self):

        self.netc.zero_grad()
        self.loss_total_real = None
        self.loss_total_fake = None

        feature, loc, classifiear = self.netc(self.img_real)
        loc_lable = self.max_act(feature)

        weak_loc = self.act_label(loc_lable)

        self.err_c_loc = (l2_loss(loc, loc_lable) + l2_loss(loc, weak_loc))


        for i in range(len(classifiear)):
            o_real = classifiear[i].unsqueeze(0)
            # o_fake = output_fake[i].unsqueeze(0)
            label = self.label_real[i].unsqueeze(0)
            # print(o_real.shape)



            loss_this_real = self.loss_func(o_real, label)
            # loss_this_fake = self.loss_func(o_fake, label)

            self.loss_total_real = loss_this_real if (self.loss_total_real is None) else (self.loss_total_real + loss_this_real)
            # self.loss_total_fake = loss_this_fake if (self.loss_total_fake is None) else (self.loss_total_fake + loss_this_fake)

        # self.err_c_real = self.loss_func(output_real, self.label_real)
        self.err_c = self.loss_total_real + self.err_c_loc *20
        self.err_c.backward()
        self.optimizer_c.step()

    ##

    def update_netg(self):
        """
        # ============================================================ #
        # (2) Update G network: log(D(G(x)))  + ||G(x) - x||           #
        # ============================================================ #

        """
        self.netg.zero_grad()
        loss_total_fake = None
        loss_total_real = None


        # self.out_g, _ = self.netd(self.fake)
        # self.label.data.resize_(self.out_g.shape).fill_(self.real_label)
        # self.err_g_bce = self.bce_criterion(self.out_g, self.label)
        # self.fake = self.netg(self.)
        self.err_g_l1l = self.mse_criterion(self.fake, self.input_img)  # constrain x' to look like x
        # self.err_g_enc = self.l2l_criterion(self.latent_o, self.latent_i)
        feature_real, output_real = self.netc(self.img_real)
        feature_fake, output_fake = self.netc(self.img_fake)
        self.err_g_loss = self.l2l_criterion(feature_fake, feature_real)
        for i in range(len(output_real)):
            o_real = output_real[i].unsqueeze(0)
            o_fake = output_fake[i].unsqueeze(0)
            label = self.label_real[i].unsqueeze(0)
            # print(o_real.shape)

            loss_this_real = self.loss_func(o_real, label)
            loss_this_fake = self.loss_func(o_fake, label)

            loss_total_real = loss_this_real if (loss_total_real is None) else (loss_total_real + loss_this_real)
            loss_total_fake = loss_this_fake if (loss_total_fake is None) else (loss_total_fake + loss_this_fake)
        # self.err_g = self.err_g_bce + self.err_g_l1l * self.opt.w_rec + (self.loss + self.err_g_loss) * self.opt.w_enc
        # self.err_g = self.err_g_bce + (loss + self.err_g_loss) * self.opt.w_enc
        self.err_g_enc = torch.abs(loss_total_fake - loss_total_real)

        self.err_g = self.err_g_enc + (self.err_g_l1l) * self.opt.w_rec + self.err_g_loss * self.opt.w_enc

        # self.err_g = self.err_g_l1l * self.opt.w_rec + (loss_total_fake + self.err_g_loss) * self.opt.w_enc
        self.err_g.backward(retain_graph=True)
        self.optimizer_g.step()

    ##
    def argument_image_rotation_plus_fake(self, X):
        for idx in range(len(X)):
            img0 = X[idx]
            for i in range(4):
                [img, label] = rotate_img(img0, i)
                self.img_fake[idx * 4 + i] = img
                self.label_fake[idx * 4 + i] = label
                # self.label_fake[idx * 4 + i] = torch.from_numpy(label).to(self.device)

    def argument_image_rotation_plus(self, X):
        for idx in range(len(X)):
            img0 = X[idx]
            for i in range(4):
                [img,label] = rotate_img(img0,i)
                self.img_real[idx*4 + i] = img
                self.label_real[idx * 4 + i] = label


    def optimize(self):
        """ Optimize netD and netG  networks.
        """
        self.argument_image_rotation_plus(self.input)
        self.input_img = self.input.to(self.device)

        self.updata_netc()


        # self.update_netd()
        # self.update_netg()

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

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data
        fixed_input = self.fixed_input.data

        return reals, fakes, fixed ,fixed_input

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
            # self.label_r = Variable(y)


            self.optimize()
            # if self.epoch >20:
            #     self.update_dis()


            # errors = self.get_errors()
            # if self.total_steps % self.opt.save_image_freq == 0:
            #     reals, fakes, fixed , fixed_input = self.get_current_images()
            #     self.visualizer.save_current_images(self.epoch, reals, fakes, fixed )
            #     if self.opt.display:
            #         self.visualizer.display_current_images(reals, fakes, fixed,fixed_input)
        # if self.epoch >20:
        # print('Epoch %d  err_c %f' % (self.epoch, self.loss_total_real.item()))
        print('Epoch %d  err_c %f err_loc %f' % (self.epoch, self.loss_total_real.item(), self.err_c_loc.item()))
        # else:
        #     print('Epoch %d  err_c %f ' % (self.epoch, self.loss_total_real.item()))
        # print('Epoch %d  err_g %f err_c %f err_c_real %f err_c_fake %f err_dis %f' % (self.epoch, self.err_g.item(),self.err_c.item(),
        #                                                                               self.loss_total_real.item(), self.loss_total_fake.item(),self.err_c_dis.item()))
        # print('' % (self.loss_total_real.item(), self.loss_total_fake.item(),self.err_c_dis.item()))
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
        self.err_c_loc = 0.1


        # Train for niter epochs.
        print(">> Training model %s." % self.name())
        train_data, test_data = cifa10Data(self.opt.normalclass)
        self.train_loader = DataLoader(train_data, batch_size=self.opt.batchsize, shuffle=True, num_workers=0, pin_memory=True)
        self.test_loader = DataLoader(test_data, batch_size=self.opt.batchsize, shuffle=False, num_workers=0, pin_memory=True)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_step()


            if self.epoch%20 == 0:
                self.test_class()
                # self.test_1()

            # self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name())
        self.test_class()
        # self.test_1()



    ##
    def test_class(self):
        with torch.no_grad():

            self.opt.load_weights = True
            print('test')
            label = torch.zeros(size=(10000,), dtype=torch.long, device=self.device)
            pre = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)
            pre_real = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)
            self.label_r = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

            self.relation = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)
            self.distance = torch.zeros(size=(40000,), dtype=torch.float32, device=self.device)
            self.relation_img = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)
            self.distance_img = torch.zeros(size=(10000,), dtype=torch.float32, device=self.device)
            self.classifiear = torch.zeros(size=(40000, 4), dtype=torch.float32, device=self.device)
            self.loc = torch.zeros(size=(40000, 2), dtype=torch.float32, device=self.device)
            self.opt.phase = 'test'

            for i, (x, y, z) in enumerate(self.test_loader):
                self.input = Variable(x)
                self.label_r = Variable(z)

                # input_img = self.input.to(self.device)


                self.argument_image_rotation_plus(self.input)

                # self.input = self.input.to(self.device)
                self.label_r = self.label_r.to(self.device)

                # self.fake = self.netg(input_img)
                # self.argument_image_rotation_plus_fake(self.fake.cpu().detach())
                # feature_fake,cls_fake = self.netc(self.img_fake)
                feature_real,loc, cls_real = self.netc(self.img_real)

                # loc_label = self.max_act(feature_real)

                classfiear_real = F.softmax((cls_real), dim=1)

                self.loc[i * 64: i * 64 + 64] = loc

                # self.classifiear[i * 64: i * 64 + 64] = classfiear_real

                prediction_real = -(torch.log(classfiear_real))
                # prediction_fake = -(torch.log(classfiear_fake))

                aaaa = (prediction_real.size(0) / 4)
                aaaa = int(aaaa)
                # prediction = prediction * (-1/4)


                pre_score_real = torch.zeros(size=(aaaa,), dtype=prediction_real.dtype, device=self.device)
                # pre_score_fake = torch.zeros(size=(aaaa,), dtype=prediction_real.dtype, device=self.device)
                # pre_score = torch.zeros(size=(aaaa,), dtype=prediction_real.dtype, device=self.device)


                for k in range(aaaa):
                    pre_score_real[k] = (prediction_real[k * 4, 0] + prediction_real[k * 4 + 1, 1] +
                                         prediction_real[k * 4 + 2, 2] + prediction_real[k * 4 + 3, 3]) / 4
                    # pre_score_fake[k] = (prediction_fake[k * 4, 0] + prediction_fake[k * 4 + 1, 1] +
                    #                      prediction_fake[k * 4 + 2, 2] + prediction_fake[k * 4 + 3, 3]) / 4
                    # thre = torch.abs(pre_score_fake[k] - pre_score_real[k])
                    # mask_1 = torch.gt(thre,pre_score_real[k])
                    # mask_2 = torch.gt(thre, pre_score_fake[k])
                    #
                    #
                    # if mask_1==torch.zeros_like(mask_1) &mask_2==torch.zeros_like(mask_1):
                    #     pre_score[k] = (pre_score_real[k] + pre_score_fake[k]) / 2
                    # elif mask_1==torch.ones_like(mask_1) &mask_2==torch.zeros_like(mask_1):
                    #     pre_score[k] = pre_score_fake[k]
                    # elif mask_1==torch.zeros_like(mask_1) &mask_2==torch.ones_like(mask_1):
                    #     pre_score[k] = pre_score_real[k]



                label[i * 16: i * 16 + aaaa] = self.label_r

                # pre[i * 16: i * 16 + aaaa] = pre_score_fake

                pre_real[i * 16: i * 16 + aaaa] = pre_score_real


            # for idx in range(10000):
            #     self.distance_img[idx] = (self.distance[idx * 4] + self.distance[idx * 4 + 1] +
            #                             self.distance[idx * 4 + 2]+self.distance[idx * 4 + 3]) / 4
            # D = torch.abs(self.distance_img - pre_real)/(self.distance_img + pre_real)

            aaaa = self.loc.cpu().numpy()
            np.savetxt('./output/log.txt', aaaa)
            bbbb = label.cpu().numpy()
            np.savetxt('./output/label.txt', bbbb)


            # auc_D = evaluate(label,D, metric=self.opt.metric)

            # auc_recon  =evaluate(label,self.distance_img, metric=self.opt.metric)
            auc_c_real = evaluate(label, pre_real, metric=self.opt.metric)
            # auc_c_fake = evaluate(label, pre, metric=self.opt.metric)

            f = open('./output/testclass.txt','a',encoding='utf-8-sig',newline='\n')

            f.write('auc_c_real:' + str(auc_c_real) + '\n')
            f.close()
            # print('Train class_real ROC AUC Score: %f ' % (auc_c_real ))
            # print('test done')

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
                self.argument_image_rotation_plus(self.input)

                # self.input = self.input.to(self.device)
                self.label_rrr = self.label_rrr.to(self.device)


                size = int(self.input.size(0))
                input_1 = torch.empty(size=(size, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                      device=self.device)
                input_2 = torch.empty(size=(size, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                      device=self.device)
                input_3 = torch.empty(size=(size, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                      device=self.device)
                input_4 = torch.empty(size=(size, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                      device=self.device)


                _,classfiear_real_1 = self.netc(self.img_real)
                classfiear_real = F.softmax(classfiear_real_1, dim=1)

                prediction_real = -(torch.log(classfiear_real))
                for j in range(size):
                    input_1[j] = self.img_real[j*4]
                    input_2[j] = self.img_real[j * 4 +1]
                    input_3[j] = self.img_real[j * 4 +2]
                    input_4[j] = self.img_real[j * 4 +3]
                output_1 = self.netg(input_1)
                output_2 = self.netg(input_2)
                output_3 = self.netg(input_3)
                output_4 = self.netg(input_4)
                _,classifiear_real = self.netc(input_1)

                _,classfiear_11 = self.netc(output_1)
                _,classfiear_21 = self.netc(output_2)
                _,classfiear_31 = self.netc(output_3)
                _,classfiear_41 = self.netc(output_4)




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
                    # label_z[k] = self.label_rrr[k * 4]
                    pre_score[k] = (prediction_1[k , 0] + prediction_2[k , 1] +
                                    prediction_3[k , 2] + prediction_4[k , 3]) / 4
                    pre_score_real[k] = (prediction_real[k * 4, 0] + prediction_real[k * 4 + 1, 1] +
                                         prediction_real[k * 4 + 2, 2] + prediction_real[k * 4 + 3, 3]) / 4

                label[i * 16: i * 16 + aaaa] = self.label_rrr
                pre[i * 16: i * 16 + aaaa] = pre_score
                pre_real[i * 16: i * 16 + aaaa] = pre_score_real

            # D = pre + self.relation * 0.2
            # D_real = pre_real + self.relation * 0.2

            # aaaa = self.classifiear.cpu().numpy()
            # np.savetxt('./output/log.txt', aaaa)
            # bbbb = label.cpu().numpy()
            # np.savetxt('./output/label.txt', bbbb)

            mu = torch.mul(pre, self.relation)
            mu_real = torch.mul(pre_real, self.relation)

            # auc_mu_fake = evaluate(label, mu, metric=self.opt.metric)
            # auc_mu_real = evaluate(label, mu_real, metric=self.opt.metric)
            # auc_d_fake = evaluate(label, D, metric=self.opt.metric)
            # auc_d_real = evaluate(label, D_real, metric=self.opt.metric)
            auc_c_fake = evaluate(label, pre, metric=self.opt.metric)
            auc_c_real = evaluate(label, pre_real, metric=self.opt.metric)
            auc_r = evaluate(label, self.relation, metric=self.opt.metric)
            auc_r_img = evaluate(label, self.relation_img, metric=self.opt.metric)

            f = open('./output/test1.txt', 'a', encoding='utf-8-sig', newline='\n')

            f.write(str(auc_c_real) +'\0' + str(auc_c_fake)+'\n')
            f.close()

            # print('Train mul_real ROC AUC Score: %f  mu_fake: %f' % (auc_mu_real, auc_mu_fake))
            # print('Train add_real ROC AUC Score: %f  add_fake: %f' % (auc_d_real, auc_d_fake))
            # print('Train class_real ROC AUC Score: %f class_fake: %f' % (auc_c_real, auc_c_fake))
            #
            # print('Train recon ROC AUC Score: %f recon_img:%f' % (auc_r,auc_r_img))
            # print('test done')













