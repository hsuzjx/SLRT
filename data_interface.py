import os.path

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader
from datasets.utils.data_augmentation import *
from datasets.base_feeder import BaseFeeder


class DataInterface(L.LightningDataModule):

    def __init__(self, features_path, annotations_path, gloss_dict_path='./.tmp', ground_truth_path='./.tmp',
                 num_workers=8, batch_size=1,
                 # dataset='',
                 **kwargs):
        super().__init__()
        self.features_path = os.path.abspath(features_path)
        self.annotations_path = os.path.abspath(annotations_path)

        self.gloss_dict_path = os.path.abspath(gloss_dict_path)
        if not os.path.exists(self.gloss_dict_path):
            os.makedirs(self.gloss_dict_path)
        self.generate_gloss_dict(self.gloss_dict_path, self.annotations_path)

        self.ground_truth_path = os.path.abspath(ground_truth_path)
        if not os.path.exists(self.ground_truth_path):
            os.makedirs(self.ground_truth_path)
        self.generate_ground_truth(self.ground_truth_path, self.annotations_path)

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.gloss_dict = None

        # transforms
        self.train_transform = Compose([RandomCrop(224), RandomHorizontalFlip(0.5), ToTensor(), TemporalRescale(0.2), ])
        self.dev_transform = Compose([CenterCrop(224), ToTensor(), ])
        self.test_transform = Compose([CenterCrop(224), ToTensor(), ])

    def generate_gloss_dict(self, gloss_dict_path, annotations_path):
        if os.path.exists(os.path.join(gloss_dict_path, 'gloss_dict.npy')):
            return True

        print('Generating gloss_dict...')
        # get corpus
        train_corpus = pd.read_csv(os.path.join(annotations_path, 'train.corpus.csv'),
                                   sep='|', header=0, index_col='id')
        test_corpus = pd.read_csv(os.path.join(annotations_path, 'test.corpus.csv'),
                                  sep='|', header=0, index_col='id')
        dev_corpus = pd.read_csv(os.path.join(annotations_path, 'dev.corpus.csv'),
                                 sep='|', header=0, index_col='id')

        # get sentences
        sentences = ([s for s in train_corpus.annotation]
                     + [s for s in test_corpus.annotation]
                     + [s for s in dev_corpus.annotation])

        # get sorted glosses
        glosses = set()
        for sentence in sentences:
            for gloss in sentence.split(' '):
                glosses.add(gloss)
        if '' in glosses:
            glosses.remove('')
        glosses = sorted(list(glosses))

        # make dict
        gloss_dict = dict()
        for i in range(len(glosses)):
            gloss_dict[glosses[i]] = i + 1
        assert gloss_dict.__len__() == len(glosses)

        # save dict
        np.save(os.path.join(gloss_dict_path, 'gloss_dict.npy'), gloss_dict)

        # check dict file, if True, return True, else, return False
        return os.path.exists(os.path.join(gloss_dict_path, 'gloss_dict.npy'))

    def generate_ground_truth(self, ground_truth_path, annotations_path):
        if (os.path.exists(os.path.join(ground_truth_path, 'phoenix2014-groundtruth-train.stm'))
                and os.path.exists(os.path.join(ground_truth_path, 'phoenix2014-groundtruth-dev.stm'))
                and os.path.exists(os.path.join(ground_truth_path, 'phoenix2014-groundtruth-test.stm'))):
            return True

        print('Generating ground truth...')

        for mode in ['train', 'dev', 'test']:
            gt_path = os.path.join(ground_truth_path, f'phoenix2014-groundtruth-{mode}.stm')
            corpus = pd.read_csv(os.path.join(annotations_path, f'{mode}.corpus.csv'),
                                 sep='|', header=0, index_col='id')
            if mode == 'train':
                corpus.drop('13April_2011_Wednesday_tagesschau_default-14', axis=0, inplace=True)
            with open(gt_path, "w") as f:
                for item in corpus.iterrows():
                    f.writelines(
                        '{} 1 {} 0.0 1.79769e+308 {}\n'.format(item[0], item[1]['signer'], item[1]['annotation']))

        return (os.path.exists(os.path.join(ground_truth_path, 'phoenix2014-groundtruth-train.stm'))
                and os.path.exists(os.path.join(ground_truth_path, 'phoenix2014-groundtruth-dev.stm'))
                and os.path.exists(os.path.join(ground_truth_path, 'phoenix2014-groundtruth-test.stm')))

    def setup(self, stage=None):
        self.gloss_dict = np.load(os.path.join(self.gloss_dict_path, 'gloss_dict.npy'), allow_pickle=True).item()
        if stage == 'fit':
            self.train_dataset = BaseFeeder(features_path=self.features_path,
                                            annotations_path=self.annotations_path,
                                            gloss_dict=self.gloss_dict,
                                            mode="train",
                                            transform=self.train_transform)
            self.dev_dataset = BaseFeeder(features_path=self.features_path,
                                          annotations_path=self.annotations_path,
                                          gloss_dict=self.gloss_dict,
                                          mode="dev",
                                          transform=self.train_transform)

        if stage == 'test':
            self.test_dataset = BaseFeeder(features_path=self.features_path,
                                           annotations_path=self.annotations_path,
                                           gloss_dict=self.gloss_dict,
                                           mode="test",
                                           transform=self.train_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          collate_fn=self.train_dataset.collate_fn, pin_memory=True, drop_last=True, )

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=self.dev_dataset.collate_fn, pin_memory=True, drop_last=True, )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=self.test_dataset.collate_fn, pin_memory=True, drop_last=True, )


if __name__ == '__main__':
    data_iface = DataInterface(
        features_path='./datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px',
        annotations_path='./datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual',
        gloss_dict_path='./datasets/.tmp',
        num_workers=10,
        batch_size=2,
    )
