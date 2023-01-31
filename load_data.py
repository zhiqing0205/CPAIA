import numpy as np
from scipy.io import loadmat
import mindspore.dataset as ds


class CustomDataSet:
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.images)


def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)


def get_loader(dataset, batch_size):
    rootPath = "./data/"
    if dataset == 'wikipedia':
        valid_len = 231
        all_data = loadmat(rootPath + dataset + "/wikipedia_features.mat")

        img_train = all_data['train_img'].astype('float32')
        img_valid = all_data['test_img'].astype('float32')[0:valid_len]
        img_test = all_data['test_img'].astype('float32')[valid_len::]

        text_train = all_data['train_txt'].astype('float32')
        text_valid = all_data['test_txt'].astype('float32')[0:valid_len]
        text_test = all_data['test_txt'].astype('float32')[valid_len::]

        label_train = all_data['train_label'].astype('int64').T - 1
        label_valid = all_data['test_label'].astype('int64').T[0:valid_len] - 1
        label_test = all_data['test_label'].astype('int64').T[valid_len::] - 1

    if dataset == 'nuswide':
        valid_len = 1000
        all_data = loadmat(rootPath + dataset + "/nuswide_features.mat")

        img_train = all_data['train_img'].astype('float32')
        img_valid = all_data['test_img'].astype('float32')[0:valid_len]
        img_test = all_data['test_img'].astype('float32')[valid_len::]

        text_train = all_data['train_txt'].astype('float32')
        text_valid = all_data['test_txt'].astype('float32')[0:valid_len]
        text_test = all_data['test_txt'].astype('float32')[valid_len::]

        label_train = all_data['train_label'].astype('int64').T
        label_valid = all_data['test_label'].astype('int64').T[0:valid_len]
        label_test = all_data['test_label'].astype('int64').T[valid_len::]

    if dataset == 'xmedia':
        valid_len = 500
        all_data = loadmat(rootPath + dataset + "/xmedia_features.mat")
        img_train = all_data['I_tr_CNN'].astype('float32')
        img_valid = all_data['I_te_CNN'].astype('float32')[0:valid_len]
        img_test = all_data['I_te_CNN'].astype('float32')[valid_len::]

        text_train = all_data['T_tr_BOW'].astype('float32')
        text_valid = all_data['T_te_BOW'].astype('float32')[0:valid_len]
        text_test = all_data['T_te_BOW'].astype('float32')[valid_len::]

        label_train = all_data['trImgCat'].astype('int64') - 1
        label_valid = all_data['teImgCat'].astype('int64')[0:valid_len] - 1
        label_test = all_data['teImgCat'].astype('int64')[valid_len::] - 1

        # valid_len=250
        # all_data = loadmat(rootPath + dataset + "/xmedia_features.mat")
        # img_train = all_data['I_tr_CNN'].astype('float32')[0:2000]
        # img_valid = all_data['I_te_CNN'].astype('float32')[0:valid_len]
        # img_test = all_data['I_te_CNN'].astype('float32')[valid_len:500]
        # text_train = all_data['T_tr_BOW'].astype('float32')[0:2000]
        # text_valid = all_data['T_te_BOW'].astype('float32')[0:valid_len]
        # text_test = all_data['T_te_BOW'].astype('float32')[valid_len:500]
        # label_train = all_data['trImgCat'].astype('int64')[0:2000] - 1
        # label_valid = all_data['teImgCat'].astype('int64')[0:valid_len] - 1
        # label_test = all_data['teImgCat'].astype('int64')[valid_len:500] - 1

    label_train = ind2vec(label_train).astype(int)
    label_valid = ind2vec(label_valid).astype(int)
    label_test = ind2vec(label_test).astype(int)

    imgs = {'train': img_train, 'valid': img_valid, 'test': img_test}
    texts = {'train': text_train, 'valid': text_valid, 'test': text_test}
    labels = {'train': label_train, 'valid': label_valid, 'test': label_test}

    shuffle = {'train': False, 'valid': False, 'test': False}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['train', 'valid', 'test']}

    dataloader = {
        x: ds.GeneratorDataset(dataset[x], ["images", "texts", "labels"], shuffle=shuffle[x]) for x in
                  ['train', 'valid', 'test']
    }
    dataloader['train'] = dataloader['train'].batch(batch_size=batch_size)
    dataloader['valid'] = dataloader['valid'].batch(batch_size=batch_size)
    dataloader['test'] = dataloader['test'].batch(batch_size=batch_size)

    input_data = {}

    img_dim = img_train.shape[1]
    text_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    input_data = {

    }

    input_data['img_train'] = img_train
    input_data['text_train'] = text_train
    input_data['label_train'] = label_train
    input_data['img_valid'] = img_valid
    input_data['text_valid'] = text_valid
    input_data['label_valid'] = label_valid
    input_data['img_test'] = img_test
    input_data['text_test'] = text_test
    input_data['label_test'] = label_test
    input_data['img_dim'] = img_dim
    input_data['text_dim'] = text_dim
    input_data['num_class'] = num_class

    return dataloader, input_data
