import os
import pickle
import pandas as pd

from pipeline.utils import multi2array
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from .base_cv import BaseCrossValidation


class IMetCrossValidation(BaseCrossValidation):
    def __init__(self, training_config):
        super().__init__(training_config)
        self.data_setting = training_config['data_setting']
        self.experiment_setting = training_config['experiment_setting']

        train_dir = self.data_setting['train_dir']
        train_path = self.data_setting['train_path']
        test_dir = self.data_setting['test_dir']
        sub_path = self.data_setting['sub_path']

        sub_df = pd.read_csv(sub_path)
        train_df = pd.read_csv(train_path)

        tst_paths = [os.path.join(test_dir, img_name + '.png') for img_name in sub_df['id']]
        trn_paths = [os.path.join(train_dir, img_name + '.png') for img_name in train_df['id']]
        trn_labels = [[int(idx) for idx in lbl.split(' ')] for lbl in train_df['attribute_ids'].tolist()]

        print('[LOG] train num {}, test num {}'.format(len(trn_paths), len(tst_paths)))
        self.x, self.y = pd.Series(trn_paths), multi2array(trn_labels, class_num=self.experiment_setting['num_classes'])
        self._split()

    def _split(self):
        if os.path.exists(self.data_setting['cv_path']):
            print('[LOG] reading cv-split from {}'.format(self.data_setting['cv_path']))
            self.mskf_split = pickle.load(open(self.data_setting['cv_path'], 'rb'))
        else:
            mskf = MultilabelStratifiedKFold(n_splits=self.experiment_setting['n_folds'],
                                             random_state=self.experiment_setting['random_seed'])
            self.mskf_split = list(mskf.split(self.x, self.y))
            pickle.dump(self.mskf_split, open(self.data_setting['cv_path'], 'wb'))
            print('[LOG] dump cv-split to {}'.format(self.data_setting['cv_path']))

    def __iter__(self):
        for fold_idx, (trn_idx, val_idx) in enumerate(self.mskf_split):
            print('[LOG] val idx top 5 {}'.format(val_idx[:5]))  # fold 0: 2  6  7 12 13
            print('[LOG] fold id {}, train {} val {}'.format(fold_idx, len(trn_idx), len(val_idx)))
            trn_x, val_x = self.x.iloc[list(trn_idx)], self.x.iloc[list(val_idx)]
            trn_y, val_y = self.y[list(trn_idx), :], self.y[list(val_idx), :]
            train_, val_ = {'trn_x': list(trn_x), 'trn_y': trn_y}, {'val_x': list(val_x), 'val_y': val_y}
            yield train_, val_
