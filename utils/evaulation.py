import torch
import cv2
import numpy as np

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1, precision, recall

def evaluate_validation_set(model, dataloader):
    model.eval()
    y_pred = []
    y_true = []
    for images,labels in dataloader:
        for i in range(len(labels)):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            img = images[i].view(1, 3, 224, 224)
            with torch.no_grad():
                logps = model(img)


            ps = torch.exp(logps)
            probab = list(ps.cpu()[0])
            pred_label = probab.index(max(probab))
            y_pred.append(pred_label)
            y_true.append(labels.cpu()[i])
    f1_score, _, _ = f1_loss(torch.tensor(y_true), torch.tensor(y_pred))
    return f1_score

def calculate_accuracy(model, dataloader):
    model.eval()
    correct_count, all_count = 0, 0
    for images,labels in dataloader:
        for i in range(len(labels)):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            img = images[i].view(1, 3, 224, 224)
            with torch.no_grad():
                logps = model(img)


            ps = torch.exp(logps)
            probab = list(ps.cpu()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.cpu()[i]
            all_count += 1
            if(true_label == pred_label):
                correct_count += 1

        
    return correct_count / all_count

def return_CAM(feature_conv, weight, class_idx, reshape_size):
    size_upsample = (224, 224)
    _, nc, h, w = feature_conv.shape
    output_cams = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w)).detach().cpu().numpy()
        cam = np.matmul(weight[idx].view(-1, 48).detach().cpu().numpy(), beforeDot)
        cam = cam.reshape(reshape_size, reshape_size)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(cam_img * 255)
        output_cams.append(cv2.resize(cam_img, size_upsample))
    return output_cams