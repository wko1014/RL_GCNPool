import csv
import scipy.io as sio
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, roc_auc_score

def gradient(model, inputs, labels, rl):
    with tf.GradientTape() as tape:
        y_hat = model(inputs, rl=rl)
        loss = tf.keras.losses.binary_crossentropy(labels, y_hat, label_smoothing=.1)

    grad = tape.gradient(loss, model.trainable_variables)
    return loss, grad


def evaluate(pred, lab, one_hot=True):
    if one_hot:
        pred = np.argmax(pred, -1)
        lab = np.argmax(lab, -1)
    else:
        pred = np.round(np.squeeze(pred))
        lab = np.squeeze(lab)

    np.seterr(divide='ignore', invalid='ignore')
    tn, fp, fn, tp = confusion_matrix(lab, pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    acc = (tn + tp) / total
    sen = np.divide(tp, np.sum([tp, fn]))
    spec = np.divide(tn, np.sum([tn, fp]))
    try:
        auc = roc_auc_score(np.squeeze(lab), np.squeeze(pred))
    except:
        auc = 0
        pass
    return auc, acc, sen, spec

data_path = '/home/ko/Desktop/pycharm-2018.3.5/projects/Data/ADNI_JY/'

def load_fold_idx(fi):
    train = np.load(data_path + f'fold_index/training_idx_fold_{(fi - 1)}.npy')  # [81]
    valid = np.load(data_path + f'fold_index/validation_idx_fold_{fi - 1}.npy')  # [11]
    testi = np.load(data_path + f'fold_index/test_idx_fold_{(fi - 1)}.npy')  # [9]
    return train, valid, testi

def id_2_idx(sub_id, train, val, test):
    train_idx = []
    val_idx = []
    test_idx = []
    for i in range(len(sub_id)):
        word = sub_id[i].split('_')
        if int(word[-2]) in train:
            train_idx.append(i)
        elif int(word[-2]) in val:
            val_idx.append(i)
        elif int(word[-2]) in test:
            test_idx.append(i)

    return train_idx, val_idx, test_idx

def load_sub_ids():
    path = data_path + 'all_cn_emci_label.csv'
    subject_IDs = []
    with open(path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            subject_IDs.append(row['Id'])
    return subject_IDs


def load_all_labels():
    path = data_path + 'all_cn_emci_label.csv'
    lbls = []
    with open(path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            lbls.append(int(row['DX_Group']))
    return lbls

def load_all_samples():
    path = data_path + 'all_CN_eMCI_raw' # Gaussian normalized
    samples = sio.loadmat(path)['data']
    return samples  # [316,114,130]: samples, ROI, timepoints

def split_fold_data(raw, subids, lbls, fi):
    trid, vlid, tsid = load_fold_idx(fi)
    tri, vli, tsi = id_2_idx(subids, trid, vlid, tsid)
    train, valid, test = raw[tri,:,:], raw[vli,:,:], raw[tsi,:,:]
    lbls = np.array(lbls)
    trlbl, vllbl, tslbl = lbls[tri], lbls[vli], lbls[tsi]
    return train, trlbl, valid, vllbl, test, tslbl

def call_dataset(fold, one_hot=True):
    raw = load_all_samples()
    subids = load_sub_ids()
    lbls = load_all_labels()
    train, trlbl, valid, vllbl, test, tslbl = split_fold_data(raw, subids, lbls, fold)

    if one_hot:
        trlbl = np.eye(np.unique(trlbl, axis=0).shape[0])[trlbl]
        vllbl = np.eye(np.unique(vllbl, axis=0).shape[0])[vllbl]
        tslbl = np.eye(np.unique(tslbl, axis=0).shape[0])[tslbl]

    # Add an additional dimension for the data to make 4-dimension tensors
    train, valid, test = np.expand_dims(train, -1), np.expand_dims(valid, -1), np.expand_dims(test, -1)
    return (train, trlbl), (valid, vllbl), (test, tslbl)