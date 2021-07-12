# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


WINDOW = 101
POLY_ORDER = 1


def extract_numbers(s: str) -> list:

    l = []

    for t in s.split():
        try:
            if t.endswith(","):
                t = t[:-1]
            l.append(float(t))
        except ValueError:
            pass

    return l


class Result:

    log_files = None
    smooth = None

    gzsl = None
    seen_list = None
    unseen_list = None
    h_list = None

    d_loss = None
    g_loss = None
    real_ins_ce = None
    fake_ins_ce = None
    real_cls_ce = None
    fake_cls_ce = None
    wasserstein_d = None

    dfake = None
    dreal = None
    gp = None

    full_plot = None

    def __init__(self, log_files: list, smooth: bool = True, full_plot: bool = False):
        self.log_files = log_files
        self.smooth = smooth
        self.full_plot = full_plot

    def parse(self):
        pass

    def plot(self):

        plt.figure()

        plt.plot(self.d_loss, "#cccccc")
        plt.plot(self.g_loss, "#cccccc")

        if self.full_plot:
            plt.plot(self.dfake, "#cccccc")
            plt.plot(self.dreal, "#cccccc")
            # plt.plot(self.gp, "#cccccc")
            # plt.plot(-np.array(self.wasserstein_d), "#cccccc")
            # plt.plot(self.real_ins_ce, "#cccccc")
            # plt.plot(np.array(self.fake_ins_ce) * 0.001, "#cccccc")
            # plt.plot(self.real_cls_ce, "#cccccc")
            # plt.plot(np.array(self.fake_cls_ce) * 0.001, "#cccccc")

        if self.smooth:
            d_loss = savgol_filter(self.d_loss, WINDOW, POLY_ORDER)
            g_loss = savgol_filter(self.g_loss, WINDOW, POLY_ORDER)

            if self.full_plot:
                dfake = savgol_filter(self.dfake, WINDOW, POLY_ORDER)
                dreal = savgol_filter(self.dreal, WINDOW, POLY_ORDER)
                # gp = savgol_filter(self.gp, WINDOW, POLY_ORDER)
                # wasserstein_d = savgol_filter(self.wasserstein_d, WINDOW, POLY_ORDER)
                # real_ins_ce = savgol_filter(self.real_ins_ce, WINDOW, POLY_ORDER)
                # fake_ins_ce = savgol_filter(self.fake_ins_ce, WINDOW, POLY_ORDER)
                # real_cls_ce = savgol_filter(self.real_cls_ce, WINDOW, POLY_ORDER)
                # fake_cls_ce = savgol_filter(self.fake_cls_ce, WINDOW, POLY_ORDER)

                plt.plot(d_loss, label="critic")
                plt.plot(dfake, label="C(x~)")
                plt.plot(dreal, label="C(x)")
                # plt.plot(gp, label="GP")
                # plt.plot(-wasserstein_d, label="Wasserstein (C)")
                # plt.plot(real_ins_ce, label="instance (C)")
                # plt.plot(real_cls_ce, label="class (C)")
                plt.plot(g_loss, label="generator")
                # plt.plot(fake_ins_ce * 0.001, label="instance (G)")
                # plt.plot(fake_cls_ce * 0.001, label="class (G)")
            else:
                plt.plot(d_loss, label="critic")
                plt.plot(g_loss, label="generator")

            plt.legend()
        else:
            if self.full_plot is not None:
                plt.legend(["critic", "C(x~)", "C(x)", "generator"])
            else:
                plt.legend(["critic", "generator"])

        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("training: {}".format(self.log_files[0]))

        plt.figure()

        if self.gzsl:

            plt.plot(self.seen_list, "#cccccc")
            plt.plot(self.unseen_list, "#cccccc")
            plt.plot(self.h_list, "#cccccc")

            if self.smooth:
                seen_list = savgol_filter(self.seen_list, WINDOW, POLY_ORDER)
                unseen_list = savgol_filter(self.unseen_list, WINDOW, POLY_ORDER)
                h_list = savgol_filter(self.h_list, WINDOW, POLY_ORDER)

                plt.plot(seen_list, label="seen")
                plt.plot(unseen_list, label="unseen")
                plt.plot(h_list, label="H")

                plt.legend()
            else:
                plt.legend(["seen", "unseen", "H"])
        else:

            plt.plot(self.unseen_list, "#cccccc")

            if self.smooth:
                unseen_list = savgol_filter(self.unseen_list, WINDOW, POLY_ORDER)
                plt.plot(unseen_list, label="unseen")
                plt.legend()
            else:
                plt.legend(["unseen"])

        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.title("test: {}".format(self.log_files[0]))

        plt.show()


class ResultKeras(Result):

    def parse(self):

        self.d_loss = []
        self.g_loss = []

        self.seen_list = []
        self.unseen_list = []
        self.h_list = []

        self.gzsl = False

        for log_file in self.log_files:

            file = open(log_file)

            for line in file.readlines():

                if "main epoch" in line:
                    numbers = extract_numbers(line)
                    dloss = numbers[1]
                    gloss = numbers[2]
                    self.d_loss.append(dloss)
                    self.g_loss.append(gloss)

                if "best acc" in line:

                    if len(line.split(" - ")) > 2:
                        self.gzsl = True

                    if self.gzsl:
                        numbers = extract_numbers(line)
                        seen = numbers[0]
                        unseen = numbers[1]
                        h = numbers[2]

                        self.seen_list.append(seen)
                        self.unseen_list.append(unseen)
                        self.h_list.append(h)
                    else:
                        unseen = extract_numbers(line)[0]
                        self.unseen_list.append(unseen)

            file.close()


class ResultTorch(Result):

    def parse(self):

        self.d_loss = []
        self.g_loss = []
        self.wasserstein_d = []
        self.real_ins_ce = []
        self.fake_ins_ce = []
        self.real_cls_ce = []
        self.fake_cls_ce = []

        self.dfake = []
        self.dreal = []
        self.gp = []

        self.seen_list = []
        self.unseen_list = []
        self.h_list = []

        self.gzsl = False

        for log_file in self.log_files:

            file = open(log_file)

            for line in file.readlines():

                if "Loss_D" in line:

                    numbers = extract_numbers(line)

                    dloss = numbers[0]
                    gloss = numbers[1]
                    wasserstein_d = numbers[2]
                    real_ins_ce = numbers[3]
                    fake_ins_ce = numbers[4]
                    real_cls_ce = numbers[5]
                    fake_cls_ce = numbers[6]
                    self.d_loss.append(dloss)
                    self.g_loss.append(gloss)
                    self.wasserstein_d.append(wasserstein_d)
                    self.real_ins_ce.append(real_ins_ce)
                    self.fake_ins_ce.append(fake_ins_ce)
                    self.real_cls_ce.append(real_cls_ce)
                    self.fake_cls_ce.append(fake_cls_ce)

                if "Dfake" in line:

                    numbers = extract_numbers(line)

                    dfake = numbers[0]
                    dreal = numbers[1]
                    gp = numbers[2]

                    self.dfake.append(dfake)
                    self.dreal.append(dreal)
                    self.gp.append(gp)

                if "class accuracy" in line:

                    unseen = extract_numbers(line)[0]
                    self.unseen_list.append(unseen)

                elif "unseen=" in line:

                    self.gzsl = True

                    numbers = extract_numbers(line)

                    unseen = numbers[0]
                    seen = numbers[1]
                    h = numbers[2]

                    self.seen_list.append(seen)
                    self.unseen_list.append(unseen)
                    self.h_list.append(h)

            file.close()


# result = ResultTorch(["log.txt"], full_plot=False)
# result.parse()
# result.plot()
