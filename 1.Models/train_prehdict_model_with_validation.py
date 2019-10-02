from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets.mnist
from torchvision import transforms
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.autograd import Variable
from sklearn.utils import class_weight

load_dataset = importlib.machinery.SourceFileLoader('load_dataset','../../0.Preprocess/load_dataset.py').load_module()
process_results = importlib.machinery.SourceFileLoader('process_results','../../2.Results/process_results.py').load_module()

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

def init_weights(m):
    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    m.bias.data.fill_(0.01)

def train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs):
    print("importing network")
    from PrehdictNetwork import PrehdictNet

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
    for train_index, test_index in skf.split(train_X, train_y):
        if args.view != "siamese":
            net = PrehdictNet(num_inputs=1, output_dim=args.output_dim).to(device)
        else:
            net = SiamNet(output_dim=args.output_dim).to(device)
        net.zero_grad()

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
            net.train()
            for batch_idx, (data, target, cov) in enumerate(training_generator):
                optimizer.zero_grad()
                output = net(data.to(device))
                target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
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
            net.eval()
            with torch.no_grad():
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
                    pred_prob = pred_prob.squeeze()
                    pred_label = torch.argmax(output, dim=1)

                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)

                    all_pred_prob_val.append(pred_prob)
                    all_targets_val.append(target)
                    all_pred_label_val.append(pred_label)

                    patient_ID_val.extend(cov)
            net.eval()
            with torch.no_grad():
                for batch_idx, (data, target, cov) in enumerate(test_generator):
                    net.zero_grad()
                    net.eval()
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
            path_to_checkpoint = args.dir + "/" + str(fold) + "_checkpoint_" + str(epoch) + '.pth'
            torch.save(checkpoint, path_to_checkpoint)
            
        fold += 1
        



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size")
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
    parser.add_argument('--unet', action="store_true", help="UNet architecthure")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--sc", default=5, type=int, help="number of skip connections for unet (0, 1, 2, 3, 4, or 5)")
    parser.add_argument("--init", default="none")
    parser.add_argument("--hydro_only", action="store_true")
    parser.add_argument("--output_dim", default=256, type=int, help="output dim for last linear layer")
    parser.add_argument("--datafile", default="../../0.Preprocess/preprocessed_images_20190617.pickle", help="File containing pandas dataframe with images stored as numpy array")
    parser.add_argument("--gender", default=None, type=str, help="choose from 'male' and 'female'")
    args = parser.parse_args()

    print("ARGS" + '\t' + str(args))

    max_epochs = args.epochs
    train_X, train_y, train_cov, test_X, test_y, test_cov = load_dataset.load_dataset(views_to_get="siamese",
                                                                                      sort_by_date=True,
                                                                                      pickle_file=args.datafile,
                                                                                      contrast=args.contrast,
                                                                                      split=args.split,
                                                                                      get_cov=True,
                                                                                      bottom_cut=args.bottom_cut,
                                                                                      etiology=args.etiology,
                                                                                      crop=args.crop, hydro_only=args.hydro_only, gender=args.gender)

    if args.view == "sag" or args.view == "trans":
        train_X_single=[]
        test_X_single=[]

        for item in train_X:
            if args.view == "sag":
                train_X_single.append(item[0])
            elif args.view == "trans":
                train_X_single.append(item[1])
        for item in test_X:
            if args.view == "sag":
                test_X_single.append(item[0])
            elif args.view == "trans":
                test_X_single.append(item[1])

        train_X=train_X_single
        test_X=test_X_single
        train_X=np.array(train_X_single)
        test_X=np.array(test_X_single)
        
        

        
    train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs)


if __name__ == '__main__':
    main()
