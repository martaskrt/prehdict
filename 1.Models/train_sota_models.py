from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.autograd import Variable
from sklearn.utils import class_weight
import sys


SEED = 42

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)

class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y, cov):
        self.X = torch.from_numpy(X).float()
        self.y = y
        self.cov = cov

    def __getitem__(self, index):
        imgs, target, cov = self.X[index], self.y[index], self.cov[index]
        return imgs, target, cov

    def __len__(self):
        return len(self.X)


def train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs):
    process_results = importlib.machinery.SourceFileLoader('process_results', args.git_dir + '/nephronetwork/2.Results/process_results.py').load_module()

    sys.path.insert(0, args.git_dir + '/1.Models/')
    if args.vgg:
        print("importing VGG16 without BN")
        from SOTAmodels import RevisedVGG
    elif args.vgg_bn:
        print("importing VGG16 with BN")
        from SOTAmodels import RevisedVGG_bn
    elif args.densenet:
        print("Importing DenseNet")
        from SOTAmodels import RevisedDenseNet
    elif args.resnet18:
        print("importing ResNet18")
        from SOTAmodels import RevisedResNet
    elif args.resnet50:
        print("importing ResNet50")
        from SOTAmodels import RevisedResNet50
    else:
        print("No model selected, add one of the following flags: --vgg, --resnet18, --resnet50")

    hyperparams = {'lr': args.lr,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay,
                   'train/test_split': args.split 
                  }
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    test_set = KidneyDataset(test_X, test_y, test_cov)

    test_generator = DataLoader(test_set, **params)

    fold=1
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_y = np.array(train_y)
    train_cov = np.array(train_cov)


    train_X, train_y, train_cov = shuffle(train_X, train_y, train_cov, random_state=42)

    if args.cv:
        for train_index, test_index in skf.split(train_X, train_y):

            if args.vgg:
                if args.view != "siamese":
                    net = RevisedVGG(num_inputs=1).to(device)
                else:
                    net = RevisedVGG(num_inputs=2).to(device)

            elif args.vgg_bn:
                if args.view != "siamese":
                    net = RevisedVGG_bn(num_inputs=1).to(device)
                else:
                    net = RevisedVGG_bn(num_inputs=2).to(device)

            elif args.densenet:
                if args.view != "siamese":
                    net = RevisedDenseNet(num_inputs=1).to(device)
                else:
                    net = RevisedDenseNet(num_inputs=2).to(device)


            elif args.resnet18:
                if args.view != "siamese":
                    net = RevisedResNet(num_inputs=1).to(device)
                else:
                    net = RevisedResNet(num_inputs=2).to(device)

            elif args.resnet50:
                if args.view != "siamese":
                    net = RevisedResNet50(num_inputs=1).to(device)
                else:
                    net = RevisedResNet50(num_inputs=2).to(device)



            if args.adam:
                optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'],
                                             weight_decay=hyperparams['weight_decay'])
            else:
                optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                            weight_decay=hyperparams['weight_decay'])


            train_X_CV = train_X[train_index]
            train_y_CV = train_y[train_index]
            train_cov_CV = train_cov[train_index]

            val_X_CV = train_X[test_index]
            val_y_CV = train_y[test_index]
            val_cov_CV = train_cov[test_index]

            print("Dataset generated")

            training_set = KidneyDataset(train_X_CV, train_y_CV, train_cov_CV)
            training_generator = DataLoader(training_set, **params)

            validation_set = KidneyDataset(val_X_CV, val_y_CV, val_cov_CV)
            validation_generator = DataLoader(validation_set, **params)

            for epoch in range(max_epochs):
                accurate_labels_train = 0
                accurate_labels_val = 0
                accurate_labels_test = 0

                loss_accum_train = 0
                loss_accum_val = 0
                loss_accum_test = 0

                all_targets_train = []
                all_pred_prob_train = []
                all_pred_label_train = []

                all_targets_val = []
                all_pred_prob_val = []
                all_pred_label_val = []

                all_targets_test = []
                all_pred_prob_test = []
                all_pred_label_test = []

                patient_ID_test = []
                patient_ID_train = []
                patient_ID_val = []


                counter_train = 0
                counter_val = 0
                counter_test = 0

                for batch_idx, (data, target, cov) in enumerate(training_generator):

                    optimizer.zero_grad()

                    output = net(data.to(device))

                    target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)

                    loss = F.cross_entropy(output, target)
                    loss_accum_train += loss.item() * len(target)

                    loss.backward()

                    accurate_labels_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                    optimizer.step()
                    counter_train += len(target)

                    output_softmax = softmax(output)
                    pred_prob = output_softmax[:, 1]
                    pred_label = torch.argmax(output_softmax, dim=1)
                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)


                    all_pred_prob_train.append(pred_prob)
                    all_targets_train.append(target)
                    all_pred_label_train.append(pred_label)

                    patient_ID_train.extend(cov)

                with torch.set_grad_enabled(False):
                    for batch_idx, (data, target, cov) in enumerate(validation_generator):
                        net.zero_grad()
                        optimizer.zero_grad()
                        output = net(data)
                        target = target.type(torch.LongTensor).to(device)

                        loss = F.cross_entropy(output, target)
                        loss_accum_val += loss.item() * len(target)
                        counter_val += len(target)
                        output_softmax = softmax(output)

                        accurate_labels_val += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                        pred_prob = output_softmax[:,1]

                        if len(pred_prob.size()) > 1:
                            pred_prob = pred_prob.squeeze()

                        pred_label = torch.argmax(output, dim=1)

                        assert len(pred_prob) == len(target)
                        assert len(pred_label) == len(target)

                        all_pred_prob_val.append(pred_prob)
                        all_targets_val.append(target)
                        all_pred_label_val.append(pred_label)

                        patient_ID_val.extend(cov)

                with torch.set_grad_enabled(False):

                    for batch_idx, (data, target, cov) in enumerate(test_generator):
                        net.zero_grad()
                        optimizer.zero_grad()
                        output = net(data)
                        target = target.type(torch.LongTensor).to(device)

                        loss = F.cross_entropy(output, target)
                        loss_accum_test += loss.item() * len(target)
                        counter_test += len(target)
                        output_softmax = softmax(output)
                        accurate_labels_test += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                        pred_prob = output_softmax[:, 1]
                        pred_prob = pred_prob.squeeze()
                        pred_label = torch.argmax(output, dim=1)

                        assert len(pred_prob) == len(target)
                        assert len(pred_label) == len(target)

                        all_pred_prob_test.append(pred_prob)
                        all_targets_test.append(target)
                        all_pred_label_test.append(pred_label)

                        patient_ID_test.extend(cov)

                all_pred_prob_train = torch.cat(all_pred_prob_train)
                all_targets_train = torch.cat(all_targets_train)
                all_pred_label_train = torch.cat(all_pred_label_train)


                assert len(all_targets_train) == len(training_set)
                assert len(all_pred_prob_train) == len(training_set)
                assert len(all_pred_label_train) == len(training_set)
                assert len(patient_ID_train) == len(training_set)

                results_train = process_results.get_metrics(y_score=all_pred_prob_train.cpu().detach().numpy(),
                                                      y_true=all_targets_train.cpu().detach().numpy(),
                                                      y_pred=all_pred_label_train.cpu().detach().numpy())

                print('Fold\t{}\tTrainEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                      'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_train)/counter_train,
                                                                            loss_accum_train/counter_train, results_train['auc'],
                                                                            results_train['auprc'], results_train['tn'],
                                                                            results_train['fp'], results_train['fn'],
                                                                            results_train['tp']))
                all_pred_prob_val = torch.cat(all_pred_prob_val)
                all_targets_val = torch.cat(all_targets_val)
                all_pred_label_val = torch.cat(all_pred_label_val)


                assert len(all_targets_val) == len(validation_set)
                assert len(all_pred_prob_val) == len(validation_set)
                assert len(all_pred_label_val) == len(validation_set)
                assert len(patient_ID_val) == len(validation_set)

                results_val = process_results.get_metrics(y_score=all_pred_prob_val.cpu().detach().numpy(),
                                                          y_true=all_targets_val.cpu().detach().numpy(),
                                                          y_pred=all_pred_label_val.cpu().detach().numpy())
                print('Fold\t{}\tValEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                      'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_val) / counter_val,
                                                                             loss_accum_val / counter_val, results_val['auc'],
                                                                             results_val['auprc'], results_val['tn'],
                                                                             results_val['fp'], results_val['fn'],
                                                                             results_val['tp']))

                all_pred_prob_test = torch.cat(all_pred_prob_test)
                all_targets_test = torch.cat(all_targets_test)
                all_pred_label_test = torch.cat(all_pred_label_test)


                assert len(all_targets_test) == len(test_y)
                assert len(all_pred_label_test) == len(test_y)
                assert len(all_pred_prob_test) == len(test_y)
                assert len(patient_ID_test) == len(test_y)

                results_test = process_results.get_metrics(y_score=all_pred_prob_test.cpu().detach().numpy(),
                                                          y_true=all_targets_test.cpu().detach().numpy(),
                                                          y_pred=all_pred_label_test.cpu().detach().numpy())
                print('Fold\t{}\tTestEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                      'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_test) / counter_test,
                                                                             loss_accum_test / counter_test, results_test['auc'],
                                                                             results_test['auprc'], results_test['tn'],
                                                                             results_test['fp'], results_test['fn'],
                                                                             results_test['tp']))


                checkpoint = {'epoch': epoch,
                              'loss': loss,
                              'hyperparams': hyperparams,
                              'args': args,
                              'model_state_dict': net.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'loss_train': loss_accum_train / counter_train,
                              'loss_val': loss_accum_val / counter_val,
                              'loss_test': loss_accum_test / counter_test,
                              'accuracy_train': int(accurate_labels_train) / counter_train,
                              'accuracy_val': int(accurate_labels_val) / counter_val,
                              'accuracy_test': int(accurate_labels_test) / counter_test,
                              'results_train': results_train,
                              'results_val': results_val,
                              'results_test': results_test,
                              'all_pred_prob_train': all_pred_prob_train,
                              'all_pred_prob_val': all_pred_prob_val,
                              'all_pred_prob_test': all_pred_prob_test,
                              'all_targets_train': all_targets_train,
                              'all_targets_val': all_targets_val,
                              'all_targets_test': all_targets_test,
                              'patient_ID_train': patient_ID_train,
                              'patient_ID_val': patient_ID_val,
                              'patient_ID_test': patient_ID_test}

                if not os.path.isdir(args.dir):
                    os.makedirs(args.dir)

            fold += 1

    else:

        if args.vgg:
            if args.view != "siamese":
                net = RevisedVGG(num_inputs=1).to(device)
            else:
                net = RevisedVGG(num_inputs=2).to(device)

        elif args.vgg_bn:
            if args.view != "siamese":
                net = RevisedVGG_bn(num_inputs=1).to(device)
            else:
                net = RevisedVGG_bn(num_inputs=2).to(device)

        elif args.densenet:
            if args.view != "siamese":
                net = RevisedDenseNet(num_inputs=1).to(device)
            else:
                net = RevisedDenseNet(num_inputs=2).to(device)

        elif args.resnet18:
                if args.view != "siamese":
                    net = RevisedResNet(num_inputs=1).to(device)
                else:
                    net = RevisedResNet(num_inputs=2).to(device)

        elif args.resnet50:
            if args.view != "siamese":
                net = RevisedResNet50(num_inputs=1).to(device)
            else:
                net = RevisedResNet50(num_inputs=2).to(device)

        if args.adam:
            optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'],
                                         weight_decay=hyperparams['weight_decay'])
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                        weight_decay=hyperparams['weight_decay'])

        train_X_CV = train_X
        train_y_CV = train_y
        train_cov_CV = train_cov

        print("Dataset generated")

        training_set = KidneyDataset(train_X_CV, train_y_CV, train_cov_CV)
        training_generator = DataLoader(training_set, **params)

        for epoch in range(args.stop_epoch):
            accurate_labels_train = 0
            accurate_labels_test = 0

            loss_accum_train = 0
            loss_accum_test = 0

            all_targets_train = []
            all_pred_prob_train = []
            all_pred_label_train = []

            all_targets_test = []
            all_pred_prob_test = []
            all_pred_label_test = []

            patient_ID_test = []
            patient_ID_train = []

            counter_train = 0
            counter_test = 0

            for batch_idx, (data, target, cov) in enumerate(training_generator):

                optimizer.zero_grad()

                output = net(data.to(device))

                target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)

                loss = F.cross_entropy(output, target)
                loss_accum_train += loss.item() * len(target)

                loss.backward()

                accurate_labels_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                optimizer.step()
                counter_train += len(target)

                output_softmax = softmax(output)
                pred_prob = output_softmax[:, 1]
                pred_label = torch.argmax(output_softmax, dim=1)
                assert len(pred_prob) == len(target)
                assert len(pred_label) == len(target)

                all_pred_prob_train.append(pred_prob)
                all_targets_train.append(target)
                all_pred_label_train.append(pred_label)

                patient_ID_train.extend(cov)

            with torch.set_grad_enabled(False):
                for batch_idx, (data, target, cov) in enumerate(test_generator):
                    net.zero_grad()
                    optimizer.zero_grad()
                    output = net(data)
                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    loss_accum_test += loss.item() * len(target)
                    counter_test += len(target)
                    output_softmax = softmax(output)
                    accurate_labels_test += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                    pred_prob = output_softmax[:, 1]
                    pred_prob = pred_prob.squeeze()
                    pred_label = torch.argmax(output, dim=1)

                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)

                    all_pred_prob_test.append(pred_prob)
                    all_targets_test.append(target)
                    all_pred_label_test.append(pred_label)

                    patient_ID_test.extend(cov)

            all_pred_prob_train = torch.cat(all_pred_prob_train)
            all_targets_train = torch.cat(all_targets_train)
            all_pred_label_train = torch.cat(all_pred_label_train)


            assert len(all_targets_train) == len(training_set)
            assert len(all_pred_prob_train) == len(training_set)
            assert len(all_pred_label_train) == len(training_set)
            assert len(patient_ID_train) == len(training_set)

            results_train = process_results.get_metrics(y_score=all_pred_prob_train.cpu().detach().numpy(),
                                                        y_true=all_targets_train.cpu().detach().numpy(),
                                                        y_pred=all_pred_label_train.cpu().detach().numpy())

            print('TrainEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(epoch,
                                                                         int(accurate_labels_train) / counter_train,
                                                                         loss_accum_train / counter_train,
                                                                         results_train['auc'],
                                                                         results_train['auprc'],
                                                                         results_train['tn'],
                                                                         results_train['fp'], results_train['fn'],
                                                                         results_train['tp']))

            all_pred_prob_test = torch.cat(all_pred_prob_test)
            all_targets_test = torch.cat(all_targets_test)
            all_pred_label_test = torch.cat(all_pred_label_test)


            assert len(all_targets_test) == len(test_y)
            assert len(all_pred_label_test) == len(test_y)
            assert len(all_pred_prob_test) == len(test_y)
            assert len(patient_ID_test) == len(test_y)

            results_test = process_results.get_metrics(y_score=all_pred_prob_test.cpu().detach().numpy(),
                                                       y_true=all_targets_test.cpu().detach().numpy(),
                                                       y_pred=all_pred_label_test.cpu().detach().numpy())
            print('TestEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(epoch,
                                                                         int(accurate_labels_test) / counter_test,
                                                                         loss_accum_test / counter_test,
                                                                         results_test['auc'],
                                                                         results_test['auprc'], results_test['tn'],
                                                                         results_test['fp'], results_test['fn'],
                                                                         results_test['tp']))
            checkpoint = {'epoch': epoch,
                          'loss': loss,
                          'hyperparams': hyperparams,
                          'args': args,
                          'model_state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_train': loss_accum_train / counter_train,
                          'loss_test': loss_accum_test / counter_test,
                          'accuracy_train': int(accurate_labels_train) / counter_train,
                          'accuracy_test': int(accurate_labels_test) / counter_test,
                          'results_train': results_train,
                          'results_test': results_test,
                          'all_pred_prob_train': all_pred_prob_train,
                          'all_pred_prob_test': all_pred_prob_test,
                          'all_targets_train': all_targets_train,
                          'all_targets_test': all_targets_test,
                          'patient_ID_train': patient_ID_train,
                          'patient_ID_test': patient_ID_test}

            if not os.path.isdir(args.dir):
                os.makedirs(args.dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument('--adam', action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("--view", default="siamese", help="siamese, sag, trans")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    parser.add_argument("--split", default=0.7, type=float, help="proportion of dataset to use as training")
    parser.add_argument("--bottom_cut", default=0.0, type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument("--etiology", default="B", help="O (obstruction), R (reflux), B (both)")
    parser.add_argument('--vgg', action='store_true',help="Run VGG16 architecture, not using this flag runs ResNet")
    parser.add_argument('--vgg_bn', action='store_true',help="Run VGG16 batch norm architecture")
    parser.add_argument('--densenet', action='store_true',help="Run DenseNet")
    parser.add_argument('--resnet18', action='store_true',help="Run ResNet18 architecture, not using this flag runs ResNet16")
    parser.add_argument('--resnet50', action='store_true',help="Run ResNet50 architecture, not using this flag runs ResNet50")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--output_dim", default=128, type=int, help="output dim for last linear layer")
    parser.add_argument("--git_dir",default="C:/Users/Lauren/Desktop/DS Core/Projects/Urology/")
    parser.add_argument('--cv', action='store_true',help="Flag to run cross validation")
    parser.add_argument("--stop_epoch", default=18, type=int, help="If not running cross validation, which epoch to finish with")

    args = parser.parse_args()

    print("batch size: " + str(args.batch_size))
    print("lr: " + str(args.lr))
    print("momentum: " + str(args.momentum))
    print("adam optimizer: " + str(args.adam))
    print("weight decay: " + str(args.weight_decay))
    print("view: " + str(args.view))


    datafile = args.git_dir + "nephronetwork/0.Preprocess/preprocessed_images_20190612.pickle"

    load_dataset_LE = importlib.machinery.SourceFileLoader('load_dataset_LE', args.git_dir + '/nephronetwork/0.Preprocess/load_dataset_LE.py').load_module()


    max_epochs = args.epochs
    train_X, train_y, train_cov, test_X, test_y, test_cov = load_dataset_LE.load_dataset(views_to_get=args.view,
                                                                                         sort_by_date=True,
                                                                                         pickle_file=datafile,
                                                                                         contrast=args.contrast,
                                                                                         split=args.split,
                                                                                         get_cov=True,
                                                                                         bottom_cut=args.bottom_cut,
                                                                                         etiology=args.etiology,
                                                                                         crop=args.crop,
                                                                                         git_dir=args.git_dir)

    train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs)


if __name__ == '__main__':
    main()


