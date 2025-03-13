import logging
import argparse
import numpy as np
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from utils.loss import BalancedBCELoss, FocalLoss
from utils.metric import roc_auc_curve
from utils.utils import setup_logger, setup_logger_benchmark, repeat_once, repeat_once_benchmark

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--device', type=str, default='cuda:1', help='cpu, cuda:0, cuda:1')
parser.add_argument('--data', type=str, default="demographic, 3bp, 2bp")
parser.add_argument('--label', type=str, default="14090", help='14090, 13080')
parser.add_argument('--mode', type=str, default="train", help='train, test')
parser.add_argument('--resample', type=str, default="None", help='None, Tomek, SMOTE, SMOTETomek')
parser.add_argument('--dim_red', type=str, default="None", help='None, PCA, kPCA, t-SNE')
parser.add_argument('--dim_num', type=int, default=6, help='3, 6, 10')
parser.add_argument('--model', type=str, default="MLP", help='SVM, MLP, TabPFN')
parser.add_argument('--loss', type=str, default="BCELoss", help='BCELoss, BalancedBCELoss, FocalLoss')
parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--repeat', type=int, default=1)
args = parser.parse_args()

dim_red = {
    "None": None, 
    "PCA": PCA(n_components=args.dim_num), 
    "kPCA": KernelPCA(n_components=args.dim_num, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1), 
    "t-SNE": TSNE(n_components=3, perplexity=30, learning_rate='auto', init='random')
}

resample = {
    "None": None, 
    "Tomek": TomekLinks(),
    "SMOTE": SMOTE(random_state=42),
    "SMOTETomek": SMOTETomek(random_state=42)
}

loss = {
    "BCELoss": nn.BCELoss(),
    "BalancedBCELoss": BalancedBCELoss(),
    "FocalLoss": FocalLoss(alpha=1, gamma=0)
}

def main():
    if args.model=="TabPFN" and args.resample in ["SMOTE", "SMOTETomek"]:
        return
    if args.loss!="BCELoss" and args.model in ["SVM", "TabPFN"]:
        return

    df = pd.read_csv("data/1120918-3BP_Baseline-and-events_MHH.csv")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    if args.data=="demographic":
        benchmark = df.iloc[:, [21, 22]]
        temp = df.iloc[:, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
        data = pd.DataFrame(imputer.fit_transform(temp))
        
    elif args.data=="3bp":
        benchmark = df.iloc[:, [21, 22]]
        temp = df.iloc[:, [12, 13, 14, 15, 16, 17, 18, 19, 20]]
        data = pd.DataFrame(imputer.fit_transform(temp))
        
    elif args.data=="2bp":
        benchmark = df.iloc[:, [21, 22]]
        temp = df.iloc[:, [12, 13, 15, 16, 18, 19]]
        data = pd.DataFrame(imputer.fit_transform(temp))
        
    else: 
        return

    if args.label=="14090":
        label = df[df.columns[1:3]]
        label_name = "O140/90D135/85"
    elif args.label=="13080":
        label = df[df.columns[5:7]]
        label_name = "O130/80D130/80"
    else:
        return
    
    count_MH = 0
    count_WCH = 0
    total_MH = 0
    total_WCH = 0
    cm_MH = np.zeros_like([[0, 0], [0, 0]])
    cm_WCH = np.zeros_like([[0, 0], [0, 0]])
    
    tprs_MH = []
    aucs_MH = []
    tprs_WCH = []
    aucs_WCH = []
    
    mean_fpr = np.linspace(0, 1, 100)
    
    all_y_pred_MH = []
    all_y_pred_WCH = []
    all_prob_MH = []
    all_prob_WCH = []

    dict_MH = {"acc": [], "ppv": [], "recall": [], "f1": [], "npv": [], "spec": []}
    dict_WCH = {"acc": [], "ppv": [], "recall": [], "f1": [], "npv": [], "spec": []}

    print(args.data, args.label, args.resample, args.dim_red, args.model, args.loss)
    for idx in range(args.repeat):
        record_MH, record_WCH, record_MH_all, record_WCH_all, output, output_all = repeat_once(args, idx, data, label, label_name, dim_red, resample, loss)
        # benchmark_MH, benchmark_WCH, benchmark_output = repeat_once_benchmark(args, idx, benchmark, label, label_name)
        for key in dict_MH.keys():
            dict_MH[key].append(record_MH[key])
            dict_WCH[key].append(record_WCH[key])

        (y_test, y_MH, y_WCH, prob_MH, prob_WCH) = output
        (y_pred_MH_all, y_pred_WCH_all, y_prob_MH_all, y_prob_WCH_all) = output_all

        all_y_pred_MH.append(y_pred_MH_all)
        all_y_pred_WCH.append(y_pred_WCH_all)
        all_prob_MH.append(y_prob_MH_all)
        all_prob_WCH.append(y_prob_WCH_all)

        Y_MH = y_test.iloc[:, 0]
        Y_WCH = y_test.iloc[:, 1]
        
        count_MH += (Y_MH==0).sum()  # 0 is MH
        count_WCH += (Y_WCH==0).sum()  # 0 is WCH
        
        total_MH += len(Y_MH)
        total_WCH += len(Y_WCH)
        
        cm_MH = cm_MH if confusion_matrix(Y_MH, y_MH) is None else confusion_matrix(Y_MH, y_MH) + cm_MH
        cm_WCH = cm_WCH if confusion_matrix(Y_WCH, y_WCH) is None else confusion_matrix(Y_WCH, y_WCH) + cm_WCH
        
        # # Compute ROC curve and AUC for MH
        fpr_MH, tpr_MH, thresholds_MH = roc_curve(Y_MH, prob_MH)
        J = tpr_MH - fpr_MH
        best_threshold = thresholds_MH[np.argmax(J)]
        # print(best_threshold)
        roc_auc_MH = auc(fpr_MH, tpr_MH)
        interp_tpr_MH = np.interp(mean_fpr, fpr_MH, tpr_MH)
        interp_tpr_MH[0] = 0.0
        tprs_MH.append(interp_tpr_MH)
        aucs_MH.append(roc_auc_MH)
        
        # # Compute ROC curve and AUC for WCH
        fpr_WCH, tpr_WCH, thresholds_WCH = roc_curve(Y_WCH, prob_WCH)
        J = tpr_WCH - fpr_WCH
        best_threshold = thresholds_WCH[np.argmax(J)]
        roc_auc_WCH = auc(fpr_WCH, tpr_WCH)
        interp_tpr_WCH = np.interp(mean_fpr, fpr_WCH, tpr_WCH)
        interp_tpr_WCH[0] = 0.0
        tprs_WCH.append(interp_tpr_WCH)
        aucs_WCH.append(roc_auc_WCH)
        
        # logger_MH = setup_logger(args, "MH", idx)
        # logger_MH.info(f'Iteration {idx}: Accuracy: {record_MH["acc"]}, Precision: {record_MH["ppv"]}, Recall: {record_MH["recall"]}, F1 Score: {record_MH["f1"]}')
        # logger_WCH = setup_logger(args, "WCH", idx)
        # logger_WCH.info(f'Iteration {idx}: Accuracy: {record_WCH["acc"]}, Precision: {record_WCH["ppv"]}, Recall: {record_WCH["recall"]}, F1 Score: {record_WCH["f1"]}')
        
        # logger_benchmark_MH = setup_logger_benchmark("MH", idx)
        # logger_benchmark_MH.info(f'Iteration {idx}: Accuracy: {benchmark_MH["acc"]}, Precision: {benchmark_MH["ppv"]}, Recall: {benchmark_MH["recall"]}, F1 Score: {benchmark_MH["f1"]}')
        # logger_benchmark_WCH = setup_logger_benchmark("WCH", idx)
        # logger_benchmark_WCH.info(f'Iteration {idx}: Accuracy: {benchmark_WCH["acc"]}, Precision: {benchmark_WCH["ppv"]}, Recall: {benchmark_WCH["recall"]}, F1 Score: {benchmark_WCH["f1"]}')
        
    avg_y_pred_MH = np.mean(all_y_pred_MH, axis=0)
    avg_y_pred_WCH = np.mean(all_y_pred_WCH, axis=0)
    avg_prob_MH = np.mean(all_prob_MH, axis=0)
    avg_prob_WCH = np.mean(all_prob_WCH, axis=0)

    from statistics import mean, stdev
    print("Evaluation matrix: ")
    for key in dict_MH.keys():
        print(key, mean(dict_MH[key]), stdev(dict_MH[key]))
        print(key, mean(dict_WCH[key]), stdev(dict_WCH[key]))

    # from utils.metric import evaluate
    # y_pred_MH = np.where(avg_y_pred_MH > 0.06, 1, 0)
    # y_pred_WCH = np.where(avg_y_pred_WCH > 0.06, 1, 0)
    # acc_MH, ppv_MH, recall_MH, f1_MH, npv_MH, specificity_MH = evaluate(label, y_pred_MH, label_name+"_MH")
    # acc_WCH, ppv_WCH, recall_WCH, f1_WCH, npv_WCH, specificity_WCH = evaluate(label, y_pred_WCH, label_name+"_WCH")
    # record_MH = {"acc": acc_MH, "ppv": ppv_MH, "recall": recall_MH, "f1": f1_MH, "npv": npv_MH, "spec": specificity_MH}
    # record_WCH = {"acc": acc_WCH, "ppv": ppv_WCH, "recall": recall_WCH, "f1": f1_WCH, "npv": npv_WCH, "spec": specificity_WCH}
    # print(record_MH)
    # print(record_WCH)
    
    df['avg_pred_MH'] = avg_y_pred_MH
    df['avg_pred_WCH'] = avg_y_pred_WCH
    df['avg_prob_MH'] = avg_prob_MH
    df['avg_prob_WCH'] = avg_prob_WCH
    df.to_excel("data/1120918-3BP_Baseline-and-events_MHH_predictions.xlsx", index=False)

    print("Coundintg")
    print(count_MH/30)
    print(count_WCH/30)
    
    print(total_MH/30)
    print(total_WCH/30)
    
    print("Confusion matrix")
    print(cm_MH/30)
    print(cm_WCH/30)
    
    roc_auc_curve(tprs_MH, aucs_MH, mean_fpr, 'MH')
    roc_auc_curve(tprs_WCH, aucs_WCH, mean_fpr, 'WCH')
    return

if __name__=="__main__":
    main()
