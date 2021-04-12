import os
import h5py
import numpy as np
import xlwt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class ResultLogger:
    def __init__(self, tag, logdir='.', verbose=False):
        super().__init__()
        self.tag = tag
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir
        self.verbose = verbose
        self.class_num = 0
        self.results = []
        # training metric
        self.training_g_loss = []
        self.training_g_c_loss = []
        self.training_g_d_loss = []
        self.training_R_loss=[]
        self.training_d_loss = []
        self.training_d_c_loss = []
        self.training_d_d_loss = []

        self.training_accuracies = []
        self.training_time = []
        # test metric
        self.test_loss = []
        self.test_d_loss = []
        self.test_c_loss = []
        self.acs_accuracies = []  # Average Class Specific Accuracy
        self.precisions = []
        self.recalls = []
        self.f_macros = []
        self.f_micros = []
        self.g_macros = []
        self.g_micros = []
        self.specificities = []
        self.accuracies_per_class = []
        self.reports = []
        self.test_time = []

    def add_test_metrics(self, y_true, y_pred, time=0.):
        self.test_time.append(time)
        y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)
        self.class_num = max(self.class_num, len(np.unique(y_true)))
        report = classification_report(y_true, y_pred, digits=5, output_dict=True)
        self.reports.append(report)

        cnf_matrix = confusion_matrix(y_true, y_pred)
        if self.verbose:
            print(cnf_matrix)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        cs_accuracy = TP / cnf_matrix.sum(axis=1)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        specificity = TN / (FP + TN)
        self.accuracies_per_class.append(cs_accuracy)
        self.acs_accuracies.append(cs_accuracy.mean())
        self.precisions.append(precision.mean())
        self.recalls.append(recall.mean())
        self.specificities.append(specificity.mean())

        f1_macro = (2 * precision * recall / (precision + recall)).mean()
        f1_micro = 2 * TP.sum() / (2 * TP.sum() + FP.sum() + FN.sum())
        self.f_macros.append(f1_macro)
        self.f_micros.append(f1_micro)

        g_marco = ((recall * specificity) ** 0.5).mean()
        g_micro = ((TP.sum() / (TP.sum() + FN.sum())) * (TN.sum() / (TN.sum() + FP.sum()))) ** 0.5
        self.g_macros.append(g_marco)
        self.g_micros.append(g_micro)

    def add_training_metrics(self, g_loss, d_loss, time=0.):
        self.training_g_loss.append(g_loss)
        self.training_d_loss.append(d_loss)
        self.training_time.append(time)

    def add_training_metrics1(self, g_loss,g_c_loss,g_d_loss,R_loss, d_loss,d_c_loss,d_d_loss, time=0.):
        self.training_g_loss.append(g_loss)
        self.training_g_c_loss.append(g_c_loss)
        self.training_g_d_loss.append(g_d_loss)
        self.training_R_loss.append(R_loss)
        self.training_d_c_loss.append(d_c_loss)
        self.training_d_d_loss.append(d_d_loss)
        self.training_d_loss.append(d_loss)
        self.training_time.append(time)

    def add_training_metrics2(self, d_loss, time=0.):
        self.training_d_loss.append(d_loss)
        self.training_time.append(time)

    def add_testing_metrics(self, test_loss, test_c_loss, test_d_loss):
        self.test_loss.append(test_loss)
        self.test_c_loss.append(test_c_loss)
        self.test_d_loss.append(test_d_loss)


    def save_prediction(self, epoch: int, labels: np.ndarray, predicts: np.ndarray, probs_0: np.ndarray,probs_1: np.ndarray,time=0.,epochs=100):
        labels = labels.astype(dtype=np.int8)
        predicts = predicts.astype(dtype=np.int8)
        self.add_test_metrics(labels, predicts, time)
        if epoch % 50 == 0 or epoch == epochs - 1:
            probs_0 = probs_0.astype(dtype=np.float32)
            probs_1 = probs_1.astype(dtype=np.float32)
            filename = self.tag + "_proba_%03d.hdf5" % epoch
            with h5py.File(self.logdir + os.sep + filename, "w") as f:
                f.create_dataset("label", data=labels)
                f.create_dataset("predict", data=predicts)
                f.create_dataset("probability_0", data=probs_0)
                f.create_dataset("probability_1", data=probs_1)
                f.attrs['epoch'] = epoch

    #for resnet-test
    def save_prediction_1(self, epoch: int, labels: np.ndarray, predicts: np.ndarray, probs_0: np.ndarray,time=0.,epochs=100):
        labels = labels.astype(dtype=np.int8)
        predicts = predicts.astype(dtype=np.int8)
        self.add_test_metrics(labels, predicts, time)
        if epoch % 25 == 0 or epoch == epochs - 1:
            probs_0 = probs_0.astype(dtype=np.float32)
            filename = self.tag + "_proba_%03d.hdf5" % epoch
            with h5py.File(self.logdir + os.sep + filename, "w") as f:
                f.create_dataset("label", data=labels)
                f.create_dataset("predict", data=predicts)
                f.create_dataset("probability_0", data=probs_0)
                f.attrs['epoch'] = epoch

    def save_prediction_2(self, labels: np.ndarray, predicts: np.ndarray,time=0.):
        labels = labels.astype(dtype=np.int8)
        predicts = predicts.astype(dtype=np.int8)
        self.add_test_metrics(labels, predicts, time)

    def save_metrics(self, use_resnet=False, only_C=False):
        # save evaluation results
        workbook = xlwt.Workbook()
        sheet1 = workbook.add_sheet('evaluation_metrics')
        sheet2 = workbook.add_sheet('evaluation_metric_per_class')
        sheet3 = workbook.add_sheet('training_test_metrics')
        titles1 = ['rec_ma', 'pre_ma', 'spe_ma', 'acsa', 'f_ma', 'f_mi', 'g_ma', 'g_mi', 'time']
        for i, title in enumerate(titles1):
            sheet1.write(0, i, title)
        for i in range(len(self.acs_accuracies)):
            row = i + 1
            sheet1.write(row, 0, self.recalls[i])
            sheet1.write(row, 1, self.precisions[i])
            sheet1.write(row, 2, self.specificities[i])
            sheet1.write(row, 3, self.acs_accuracies[i])
            sheet1.write(row, 4, self.f_macros[i])
            sheet1.write(row, 5, self.f_micros[i])
            sheet1.write(row, 6, self.g_macros[i])
            sheet1.write(row, 7, self.g_micros[i])
            sheet1.write(row, 8, self.test_time[i])

        row = 0
        for i in range(len(self.acs_accuracies)):
            titles2 = ['epoch ' + str(i), 'accuracy', 'precision', 'recall', 'f1-score', 'support']
            for j, title in enumerate(titles2):
                sheet2.write(row, j, title)
            for j in range(self.class_num):
                row += 1
                sheet2.write(row, 0, 'class ' + str(j))
                sheet2.write(row, 1, self.accuracies_per_class[i][j])
                sheet2.write(row, 2, self.reports[i][str(j)]['precision'])
                sheet2.write(row, 3, self.reports[i][str(j)]['recall'])
                sheet2.write(row, 4, self.reports[i][str(j)]['f1-score'])
                sheet2.write(row, 5, self.reports[i][str(j)]['support'])
            row += 2

        if use_resnet:
            titles3 = ['time', 'train_c_loss', 'test_c_loss']
        else:
            titles3 = ['time', 'd_loss', 'd_c20_loss', 'd_c10_loss', 'g_loss', 'g_c20_loss', 'g_c10_loss', 'R_loss','test_loss',
                       'test_c20_loss', 'test_c10_loss']
        for i, title in enumerate(titles3):
            sheet3.write(0, i, title)
        for i in range(len(self.training_d_loss)):
            row = i + 1
            if use_resnet and not only_C:
                sheet3.write(row, 0, self.training_time[i])
                sheet3.write(row, 1, self.training_d_loss[i])
                sheet3.write(row, 2, self.training_g_loss[i])
                temp_column = 2
            elif use_resnet and only_C:
                sheet3.write(row, 0, self.training_time[i])
                sheet3.write(row, 1, self.training_d_loss[i])
                temp_column = 1
            else:
                sheet3.write(row, 0, self.training_time[i])
                sheet3.write(row, 1, self.training_d_loss[i])
                sheet3.write(row, 2, self.training_d_c_loss[i])
                sheet3.write(row, 3, self.training_d_d_loss[i])
                sheet3.write(row, 4, self.training_g_loss[i])
                sheet3.write(row, 5, self.training_g_c_loss[i])
                sheet3.write(row, 6, self.training_g_d_loss[i])
                sheet3.write(row, 7, self.training_R_loss[i])
                temp_column = 7
            if not use_resnet:
                sheet3.write(row, temp_column + 1, self.test_loss[i])
                sheet3.write(row, temp_column + 2, self.test_c_loss[i])
                sheet3.write(row, temp_column + 3, self.test_d_loss[i])
            else:
                sheet3.write(row, temp_column + 1, self.test_loss[i])
        filename = self.tag + '_result' + '.xls'
        workbook.save(self.logdir + os.sep + filename)

    def reset(self):
        self.class_num = 0
        self.results.clear()
        self.acs_accuracies.clear()
        self.precisions.clear()
        self.recalls.clear()
        self.f_macros.clear()
        self.f_micros.clear()
        self.g_macros.clear()
        self.g_micros.clear()
        self.specificities.clear()
        self.accuracies_per_class.clear()
        self.reports.clear()


if __name__ == '__main__':
    rl = ResultLogger('test1')
    results = []
    for _ in range(50):
        y_pred = np.random.randint(10, size=1000)
        y_true = np.random.randint(10, size=1000)
        rl.add_test_metrics(y_true, y_pred)
    rl.save_metrics()
