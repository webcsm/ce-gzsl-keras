# -*- coding: utf-8 -*-

import numpy as np

import scipy.io
import os
from sklearn.preprocessing import MinMaxScaler
import logging


logging.basicConfig(level=logging.INFO)


class Dataset:

    attribute = None
    train_feature = None
    train_label = None
    dataset_folder = None
    seen_class = None
    unseen_class = None

    test_unseen_feature = None
    test_unseen_label = None
    test_seen_feature = None
    test_seen_label = None

    def __init__(self, folder: str):
        self.dataset_folder = folder

    def read(self, dataset_name: str, preprocessing: bool = False, validation: bool = False):

        matcontent = scipy.io.loadmat(os.path.join(self.dataset_folder, dataset_name, "res101.mat"))
        feature = matcontent['features'].T
        # all_file = matcontent['image_files']
        label = matcontent['labels'].astype(int).squeeze() - 1

        matcontent = scipy.io.loadmat(os.path.join(self.dataset_folder, dataset_name, "att_splits.mat"))
        # numpy array index starts from 0, matlab starts from 1

        trainval_loc = matcontent['trainval_loc'].squeeze() - 1

        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1

        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        if matcontent.get("val_seen_loc") is not None:
            val_seen_loc = matcontent['val_seen_loc'].squeeze() - 1
            val_unseen_loc = matcontent['val_unseen_loc'].squeeze() - 1
        else:
            val_seen_loc = None

            logging.info("This dataset does not support GZSL validation")

        if not validation:

            trloc = trainval_loc
            tsloc = test_seen_loc
            tuloc = test_unseen_loc

            if preprocessing:
                scaler = MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trloc])
                _test_seen_feature = scaler.transform(feature[tsloc])
                _test_unseen_feature = scaler.transform(feature[tuloc])
            else:
                _train_feature = feature[trloc]
                _test_seen_feature = feature[tsloc]
                _test_unseen_feature = feature[tuloc]

            self.test_seen_feature = _test_seen_feature
            self.test_seen_label = label[tsloc]

        else:

            trloc = train_loc
            tsloc = val_seen_loc
            tuloc = val_unseen_loc

            if preprocessing:
                scaler = MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trloc])
                if val_seen_loc is not None:
                    _test_seen_feature = scaler.transform(feature[tsloc])
                _test_unseen_feature = scaler.transform(feature[tuloc])
            else:
                _train_feature = feature[trloc]
                if val_seen_loc is not None:
                    _test_seen_feature = feature[tsloc]
                _test_unseen_feature = feature[tuloc]

            if val_seen_loc is not None:
                self.test_seen_feature = _test_seen_feature
                self.test_seen_label = label[tsloc]

        self.attribute = matcontent['att'].T
        self.train_feature = _train_feature
        self.train_label = label[trloc]

        self.test_unseen_feature = _test_unseen_feature
        self.test_unseen_label = label[tuloc]

        self.unseen_class = np.unique(self.test_unseen_label)
        self.seen_class = np.unique(self.train_label)

        logging.info("preprocessing: {} - validation: {}".format(preprocessing, validation))
        logging.info("features: {} - attributes: {}".format(feature.shape, self.attribute.shape))
        logging.info("seen classes: {} - unseen classes: {}".format(len(self.seen_class), len(self.unseen_class)))

        test_seen_count = 0
        if self.test_seen_label is not None:
            test_seen_count = len(self.test_seen_label)

        logging.info("training: {} - test seen: {} - test unseen: {}".format(len(self.train_label), test_seen_count, len(self.test_unseen_label)))

    def attributes(self):
        return self.attribute.astype(np.float32)

    def unseen_classes(self):
        return self.unseen_class.astype(np.int32)

    def seen_classes(self):
        return self.seen_class.astype(np.int32)

    def attribute_size(self):
        return self.attribute.shape[1]

    def feature_size(self):
        return self.train_feature.shape[1]

    def train_features(self):
        return self.train_feature.astype(np.float32)

    def train_attributes(self):
        return self.attribute[self.train_label].astype(np.float32)

    def train_labels(self):
        return self.train_label.astype(np.int32)

    def attribute_seen(self):
        return self.attribute[self.seen_class].astype(np.float32)

    def test_unseen_features(self):
        return self.test_unseen_feature.astype(np.float32)

    def test_unseen_labels(self):
        return self.test_unseen_label.astype(np.int32)

    def test_seen_features(self):
        assert self.test_seen_feature is not None, "Validation set does not support GZSL"
        return self.test_seen_feature.astype(np.float32)

    def test_seen_labels(self):
        assert self.test_seen_label is not None, "Validation set does not support GZSL"
        return self.test_seen_label.astype(np.int32)
