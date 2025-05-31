import numpy as np

def accuracy_metric(predictions, ground_truth):
    correct = 0
    for index, pred  in predictions:
        if (pred != 0 and ground_truth[index] != 0) or (pred == 0 and ground_truth[index] == 0):
            correct += 1
    return correct / len(predictions) if predictions else 0.0

def calculate_f_n(predictions, ground_truth):
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for index, pred  in predictions:
        if (pred != 0 and ground_truth[index] != 0):
            tp += 1
        elif (pred == 0 and ground_truth[index] != 0):
            fn += 1
        elif (pred != 0 and ground_truth[index] == 0):
            fp += 1 
        elif (pred == 0 and ground_truth[index] == 0):
            tn += 1
    return tp, fp, fn, tn

def precision_metric(predictions, ground_truth, ):
    tp, fp, fn, tn = calculate_f_n(predictions, ground_truth)
    precission = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precission

def f1_metric(predictions, ground_truth, ):
    tp, fp, fn, tn = calculate_f_n(predictions, ground_truth)
    precission = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2*precission*recall)/(precission+recall) if (precission+recall) > 0 else 0.0
    return f1

def confusion_matrix(predictions, ground_truth):
    tp, fp, fn, tn = calculate_f_n(predictions, ground_truth)
    return [f'tp: {tp}', f'fp: {fp}', f'fn: {fn}', f'tn: {tn}']    
