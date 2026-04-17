import numpy as np
from skimage import io, transform
import os
from pyflim import util
from sklearn.metrics import auc, f1_score, mean_absolute_error
from scipy.spatial.distance import dice
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import correlate
from matplotlib import pyplot as plt
import sys

eps = sys.float_info.epsilon
class FLIMMetrics():
    def __init__(self, use_smeasure=False):
        #detection
        self.computed_detection_metrics = False
        self.pre = dict()
        self.rec = dict()
        self.pr_curve = None
        self.fscore = dict()
        self.ap = dict()
        self.mu_ap = None
        self.iou_thresholds = [item/100 for item in range(30, 96, 5)]
        self.total_components = 0
        self.correct_comp_found = dict()
        self.wrong_comp_found = dict()
        self.missed_comp = dict()
        self.cumulative_prec = dict()
        self.cumulative_rec = dict()
        self.use_smeasure = False

        #saliency
        self.computed_saliency_metrics = False
        self.wfscore = []
        self.mae = []
        self.smeasure = []
        self.emeasure = []
        

    def print_results(self, mode="readable", method="Flim"):
        if(mode == "readable"):
            if(self.computed_detection_metrics and not self.computed_saliency_metrics):
                print("Method       |  uF-M⁵| mAP⁵  | uF-M⁷⁵| mAP⁷⁵ | muAP  ")
                print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f}".
                  format(method, self.fscore["0.5"], self.ap["0.5"], self.fscore["0.75"],self.ap["0.75"],self.mu_ap))
            elif(not self.computed_detection_metrics and self.computed_saliency_metrics):
                if(self.use_smeasure):
                    print("Method       | uWF-M |   MAE |  S-M  |  E-M  | DICE ")
                    print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f}".
                      format(method, self.wfscore.mean(), self.mae.mean(), self.smeasure.mean(), self.emeasure.mean(), self.dices.mean()))
                else:
                    print("Method       | uWF-M |   MAE |  E-M  | DICE ")
                    print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f}".
                      format(method, self.wfscore.mean(), self.mae.mean(), self.emeasure.mean(), self.dices.mean()))
            elif(self.computed_detection_metrics and self.computed_saliency_metrics):
                if(self.use_smeasure):
                    print("Method       |  uF-M⁵| mAP⁵  | uF-M⁷⁵| mAP⁷⁵ | muAP  | uWF-M |   MAE |  S-M  |  E-M  | DICE ")
                    print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} | {6:.3f} | {7:.3f} | {8:.3f} | {9:.3f} | {10:.3f}".
                      format(method, self.fscore["0.5"], self.ap["0.5"], self.fscore["0.75"],self.ap["0.75"],self.mu_ap, self.wfscore.mean(), self.mae.mean(), self.smeasure.mean(), self.emeasure.mean(), self.dices.mean()))
                else:
                    print("Method       |  uF-M⁵| mAP⁵  | uF-M⁷⁵| mAP⁷⁵ | muAP  | uWF-M |   MAE |  E-M  | DICE ")
                    print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} | {6:.3f} | {7:.3f} | {8:.3f} | {9:.3f}".
                      format(method, self.fscore["0.5"], self.ap["0.5"], self.fscore["0.75"],self.ap["0.75"],self.mu_ap, self.wfscore.mean(), self.mae.mean(), self.emeasure.mean(), self.dices.mean()))
            else:
                print("No metric has been computed yet")
        elif(mode == "latex"):
            print("Latex")
        else:
            assert False, mode+" is an undefined mode"
            
    def save_results(self, output_file, mode="readable", method="Flim"):
        if(mode == "readable"):
            with open(output_file, 'w') as f:
                if(self.computed_detection_metrics and not self.computed_saliency_metrics):
                    print("Method       |  uF-M⁵| mAP⁵  | uF-M⁷⁵| mAP⁷⁵ | muAP  ", file=f)
                    print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f}".
                      format(method, self.fscore["0.5"], self.ap["0.5"], self.fscore["0.75"],self.ap["0.75"],self.mu_ap), file=f)
                elif(not self.computed_detection_metrics and self.computed_saliency_metrics):
                    if(self.use_smeasure):
                        print("Method       | uWF-M |   MAE |  S-M  |  E-M  | DICE ", file=f)
                        print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f}".
                          format(method, self.wfscore.mean(), self.mae.mean(), self.smeasure.mean(), self.emeasure.mean(), self.dices.mean()), file=f)
                    else:
                        print("Method       | uWF-M |   MAE |  E-M  | DICE ", file=f)
                        print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f}".
                          format(method, self.wfscore.mean(), self.mae.mean(), self.emeasure.mean(), self.dices.mean()), file=f)
                elif(self.computed_detection_metrics and self.computed_saliency_metrics):
                    if(self.use_smeasure):
                        print("Method       |  uF-M⁵| mAP⁵  | uF-M⁷⁵| mAP⁷⁵ | muAP  | uWF-M |   MAE |  S-M  |  E-M  | DICE ", file=f)
                        print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} | {6:.3f} | {7:.3f} | {8:.3f} | {9:.3f}".
                          format(method, self.fscore["0.5"], self.ap["0.5"], self.fscore["0.75"],self.ap["0.75"],self.mu_ap, self.wfscore.mean(), self.mae.mean(), self.smeasure.mean(), self.emeasure.mean(), self.dices.mean()), file=f)
                    else:
                        print("Method       |  uF-M⁵| mAP⁵  | uF-M⁷⁵| mAP⁷⁵ | muAP  | uWF-M |   MAE |  E-M  | DICE ", file=f)
                        print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} | {6:.3f} | {7:.3f} | {8:.3f}".
                          format(method, self.fscore["0.5"], self.ap["0.5"], self.fscore["0.75"],self.ap["0.75"],self.mu_ap, self.wfscore.mean(), self.mae.mean(), self.emeasure.mean(), self.dices.mean()), file=f)
        elif(mode == "latex"):
            print("Latex")
            
    def append_results(self, output_file, header="", mode="readable", method="Flim"):
        if(mode == "readable"):
            with open(output_file, 'a') as f:
                if(header == ""):
                    print("Setup: Unchanged BBs", file=f)
                else:
                    print("Setup: ", header, file=f)
                if(self.computed_detection_metrics and not self.computed_saliency_metrics):
                    print("Method       |  uF-M⁵| mAP⁵  | uF-M⁷⁵| mAP⁷⁵ | muAP  ", file=f)
                    print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f}".
                      format(method, self.fscore["0.5"], self.ap["0.5"], self.fscore["0.75"],self.ap["0.75"],self.mu_ap), file=f)
                elif(not self.computed_detection_metrics and self.computed_saliency_metrics):
                    if(self.use_smeasure):
                        print("Method       | uWF-M |   MAE |  S-M  |  E-M  | DICE ", file=f)
                        print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f}".
                          format(method, self.wfscore.mean(), self.mae.mean(), self.smeasure.mean(), self.emeasure.mean(), self.dices.mean()), file=f)
                    else:
                        print("Method       | uWF-M |   MAE |  E-M  | DICE ", file=f)
                        print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f}".
                          format(method, self.wfscore.mean(), self.mae.mean(), self.emeasure.mean(), self.dices.mean()), file=f)
                elif(self.computed_detection_metrics and self.computed_saliency_metrics):
                    if(self.use_smeasure):
                        print("Method       |  uF-M⁵| mAP⁵  | uF-M⁷⁵| mAP⁷⁵ | muAP  | uWF-M |   MAE |  S-M  |  E-M  | DICE ", file=f)
                        print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} | {6:.3f} | {7:.3f} | {8:.3f} | {9:.3f} | {10:.3f}".
                          format(method, self.fscore["0.5"], self.ap["0.5"], self.fscore["0.75"],self.ap["0.75"],self.mu_ap, self.wfscore.mean(), self.mae.mean(), self.smeasure.mean(), self.emeasure.mean(), self.dices.mean()), file=f)
                    else:
                        print("Method       |  uF-M⁵| mAP⁵  | uF-M⁷⁵| mAP⁷⁵ | muAP  | uWF-M |   MAE |  E-M  | DICE ", file=f)
                        print("{0:<13}| {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} | {6:.3f} | {7:.3f} | {8:.3f} | {9:.3f}".
                          format(method, self.fscore["0.5"], self.ap["0.5"], self.fscore["0.75"],self.ap["0.75"],self.mu_ap, self.wfscore.mean(), self.mae.mean(), self.emeasure.mean(), self.dices.mean()), file=f)
        elif(mode == "latex"):
            print("Latex")

    def get_total_components(self, label_folder, file_list):
        total_components = 0
        label_ext = os.listdir(label_folder)[0].split(".")[1] #util.get_files_extension(label_folder) 
        for i,filename in enumerate(file_list):
            filename = filename.split(".")[0]
            label_bbs, label_shape = util.load_bb(label_folder+"/"+filename,label_ext)
            
            total_components+=len(label_bbs)
        self.total_components = total_components

    def compare_components_detection(label_bbs, detected_bbs, iou_threshold=0.33):
        comp_found = 0
        for label_bb in label_bbs:
            for bb in detected_bbs:
                iou = FLIMMetrics.bb_intersection_over_union(label_bb, bb)
                if(iou > iou_threshold):
                    comp_found += 1
                    break
        
        return comp_found, ((len(label_bbs)) - comp_found), (len(detected_bbs) - comp_found)

    def bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def get_fscores(self):
        assert "0.5" in self.correct_comp_found and "0.75" in self.correct_comp_found, "Run 'evaluate metrics before this'"

        thresholds = ["0.5", "0.75"]
        for t in thresholds:
            tp = self.correct_comp_found[t]
            fp = self.wrong_comp_found[t]
            fn = self.missed_comp[t]
            if(tp+fp > 0.0 and tp+fn > 0.0):
                if(tp+fp > 0.0):
                    pre = tp/(tp+fp)
                elif(fp > 0.0):
                    pre = 0.0
                else:
                    pre = 1.0
                if(tp + fn > 0.0):
                    rec = tp/(tp+fn)
                else:
                    rec = 1.0
                beta = 2.0
                if(pre == 0.0 and rec == 0.0):
                    self.fscore[t] = 0
                else:
                    self.fscore[t] = (1+beta**2)*(pre*rec)/((beta**2*pre)+rec)
            elif(tp+fn == 0.0 and fp == 0.0):
                self.fscore[t] = 1.0
            else:
                self.fscore[t] = 0.0

    def get_aps(self):
        AP = 0.0
        for t in self.iou_thresholds:
            self.cumulative_rec[str(t)].append(0.0)
            self.cumulative_prec[str(t)].append(1.0)
            pre_inverted = self.cumulative_prec[str(t)][::-1]
            pre_interp = pre_inverted.copy()
            recall = self.cumulative_rec[str(t)]
            recall.sort()
            pre_interp.sort()
            AP += auc(recall, pre_interp)
        self.mu_ap = AP/len(self.iou_thresholds)

        thresholds = ["0.5", "0.75"]
        for t in thresholds:
            pre_inverted = self.cumulative_prec[t][::-1]
            pre_interp = pre_inverted.copy()
            recall = self.cumulative_rec[t]
            pre_interp.sort()
            recall.sort()
            self.ap[t] = auc(recall, pre_interp)
        

    def get_pr_curve(self):
        precision = []
        recall = []
        for t in self.iou_thresholds:
            tp = self.correct_comp_found[str(t)]
            fp = self.wrong_comp_found[str(t)]
            fn = self.missed_comp[str(t)]
            if(tp+fp > 0.0):
                pre = tp/(tp+fp)
            elif(fp > 0.0):
                pre = 0.0
            else:
                pre = 1.0
            if(tp + fn > 0.0):
                rec = tp/(tp+fn)
            else:
                rec = 1.0
            precision.append(pre)
            recall.append(rec)
        recall = recall[::-1]
        self.pr_curve = [precision, recall]

    def show_pr_curve(self, color="red", model_name="flim", title="Test"):
        assert self.pr_curve != None, "Please compute the PR curve before asking to show"

        fig, ax = plt.subplots()
        ax.plot(self.pr_curve[1],self.pr_curve[0], color="red", label=model_name)
        ax.set_title(title)
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.legend()
        
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        
        plt.show()
        plt.close()

    def save_pr_curve(self, output_file, color="red", model_name="flim", title="Test"):
        assert self.pr_curve != None, "Please compute the PR curve before asking to show"

        fig, ax = plt.subplots()
        ax.plot(self.pr_curve[1], self.pr_curve[0], color="red", label=model_name)
        ax.set_title(title)
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.legend()
        
        plt.savefig(output_file)
        plt.close()

    def dice_score(true, pred):
        """
        Dice score computation.
        Args: 
            true: binary mask of ground thruth 
            pred: binary mask of predictions
        Return:
            dice score
        """
        
        true = np.copy(true)
        pred = np.copy(pred)
        true[true > 0] = 1
        pred[pred > 0] = 1
        inter = true * pred
        denom = true + pred
        if(true.sum() == 0 and pred.sum() > 0):
            return 0.0
        elif(true.sum() > 0 and pred.sum() == 0):
            return 0.0
        elif(true.sum() == 0 and pred.sum() == 0):
            return 1.0
        return 2.0 * np.sum(inter) / np.sum(denom)

    def object_score(saliency, label):
        u = np.mean(saliency[label == 1])
        sigma = np.std(saliency[label == 1])
        return 2.0 * u / (u ** 2 + 1.0 + sigma + eps)

    def s_object(saliency, label):
        # foreground object score
        saliency_fg = saliency.copy()
        saliency_fg[label == 0] = 0
        fg_score = FLIMMetrics.object_score(saliency_fg, label)
        # background object score
        saliency_bg = 1.0 - saliency.copy()
        saliency_bg[label == 1] = 0
        bg_score = FLIMMetrics.object_score(saliency_bg, label == 0)
    
        u = np.mean(label)
        return u * fg_score + (1 - u) * bg_score

    def ssim(saliency, label):
        label_ = label.astype(np.float32)
    
        height, width = saliency.shape
        num_pixels = width * height
    
        # Compute means
        u_saliency = np.mean(saliency)
        u_label = np.mean(label_)
    
        # Compute variances
        sigma_x2 = np.sum(np.sum((saliency - u_saliency) ** 2)) / (num_pixels - 1 + eps)
        sigma_y2 = np.sum(np.sum((label_ - u_label) ** 2)) / (num_pixels - 1 + eps)
    
        # Compute covariance
        sigma_xy = np.sum(np.sum((saliency - u_saliency) * (label_ - u_label))) / (num_pixels - 1 + eps)
    
        alpha = 4 * u_saliency * u_label * sigma_xy
        beta = (u_saliency ** 2 + u_label ** 2) * (sigma_x2 + sigma_y2)
    
        if alpha != 0:
            ssim_value = alpha / (beta + eps)
        elif alpha == 0 and beta == 0:
            ssim_value = 1.0
        else:
            ssim_value = 0
    
        return ssim_value

    def split_label_with_weights(label, x, y):
        height, width = label.shape
        area = width * height
    
        # copy the 4 regions
        lt = label[:y, :x]
        rt = label[:y, x:]
        lb = label[y:, :x]
        rb = label[y:, x:]
    
        # Find the weights
        w1 = (x * y) / area
        w2 = ((width - x) * y) / area
        w3 = (x * (height - y)) / area
        w4 = 1.0 - w1 - w2 - w3
    
        return lt, rt, lb, rb, w1, w2, w3, w4

    def split_saliency(saliency, x, y):
        # copy the 4 regions
        lt = saliency[:y, :x]
        rt = saliency[:y, x:]
        lb = saliency[y:, :x]
        rb = saliency[y:, x:]
    
        return lt, rt, lb, rb

    def get_centroid(label):
        rows, cols = label.shape
    
        if np.sum(label) == 0:
            x = np.round(cols / 2)
            y = np.round(rows / 2)
        else:
            total = np.sum(label)
            i = np.arange(cols).reshape(1, cols) + 1
            j = np.arange(rows).reshape(rows, 1) + 1
    
            x = int(np.round(np.sum(np.sum(label, 0, keepdims=True) * i) / total))
            y = int(np.round(np.sum(np.sum(label, 1, keepdims=True) * j) / total))
    
        return x, y

    def s_region(saliency, label):
        x, y = FLIMMetrics.get_centroid(label)
        gt_1, gt_2, gt_3, gt_4, w1, w2, w3, w4 = FLIMMetrics.split_label_with_weights(label, x, y)
    
        sm_1, sm_2, sm_3, sm_4 = FLIMMetrics.split_saliency(saliency, x, y)
    
        q1 = FLIMMetrics.ssim(sm_1, gt_1)
        q2 = FLIMMetrics.ssim(sm_2, gt_2)
        q3 = FLIMMetrics.ssim(sm_3, gt_3)
        q4 = FLIMMetrics.ssim(sm_4, gt_4)
    
        region_value = w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4
    
        return region_value

    def s_measure(sal, gt):
        if(sal.max() > 0.0):
            saliency = sal.astype(np.float32)/sal.max()
        else:
            saliency = sal.astype(np.float32)
        if(gt.max() > 0.0):
            label = gt.astype(np.float32)
            label[label > 0.0] = 1.0
        else:
            label = gt.astype(np.float32)
        u_label = np.mean(label)
    
        if u_label == 0:  # if the GT is completely black
            u_saliency = np.mean(saliency)
            measure = 1.0 - u_saliency  # only calculate the area of intersection
        elif u_label == 1:  # if the GT is completely white
            u_saliency = np.mean(saliency)
            measure = u_saliency.copy()  # only calcualte the area of intersection
        else:
            alpha = 0.5
            measure = alpha * FLIMMetrics.s_object(saliency, label) + (1 - alpha) * FLIMMetrics.s_region(saliency, label)
            if measure < 0:
                measure = 0

        return measure
    
    #adapted from https://github.com/Mehrdad-Noori/Saliency-Evaluation-Toolbox
    def adaptive_binary(sal):
        adaptive_threshold = 2 * np.mean(sal)

        if adaptive_threshold > 1:
            adaptive_threshold = 1

        binary_sal = (sal >= adaptive_threshold).astype(np.float32)

        return binary_sal
    
    #adapted from https://github.com/Mehrdad-Noori/Saliency-Evaluation-Toolbox
    def alignment_term(dgt, dsm):
        mu_fm = np.mean(dsm)
        mu_gt = np.mean(dgt)

        align_fm = dsm - mu_fm
        align_gt = dgt - mu_gt

        return  2 * (align_gt * align_fm) / (align_gt*align_gt + align_fm*align_fm + eps)
    
    #adapted from https://github.com/Mehrdad-Noori/Saliency-Evaluation-Toolbox
    def enhanced_aligment_term(align_matrix):
        return ((align_matrix + 1) ** 2) / 4
    
    #adapted from https://github.com/Mehrdad-Noori/Saliency-Evaluation-Toolbox
    def e_measure(sal, gt):
        sm = FLIMMetrics.adaptive_binary(sal)

        gt = gt.astype(bool)
        sm = sm.astype(bool)

        dgt = gt.astype(np.float32)
        dsm = sm.astype(np.float32)

        if np.sum(dgt) == 0:
            enhanced_matrix = 1.0 - dsm
        elif np.mean(dgt) == 1:
            enhanced_matrix = dsm
        else:
            align_matrix = FLIMMetrics.alignment_term(dsm, dgt)
            enhanced_matrix = FLIMMetrics.enhanced_aligment_term(align_matrix)

        h, w = gt.shape
        return np.sum(enhanced_matrix) / (h * w - 1 + eps)

    def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def weighted_fmeasure(sal, gt, beta2=2):
        if(sal.max() > 0.0):
            saliency = sal.astype(np.float32)/sal.max()
        else:
            saliency = sal.astype(np.float32)
        if(gt.max() > 0.0):
            label = gt.astype(np.float32)
            label[label > 0.0] = 1.0
        else:
            label = gt.astype(np.float32)
        if(label.sum() == 0.0):
            if(saliency.sum() == 0.0):
                prec = 1.0
                rec = 1.0
            else:
                prec = 0.0
                rec = 1.0
        elif(saliency.sum() == 0.0):
            prec = 1.0
            rec = 0.0
        else:
            dst, idx = distance_transform_edt(1 - label, return_indices=True)
        
            raw_idx = idx[0][label == 0]
            col_idx = idx[1][label == 0]
        
            e = np.abs(saliency - label).astype(np.float32)
            et = np.abs(saliency - label).astype(np.float32)
        
            et[label == 0] = et[raw_idx, col_idx]
        
            k = FLIMMetrics.matlab_style_gauss2d(shape=(7, 7), sigma=5)
        
            ea = correlate(et.astype(np.float32), k, mode='constant')
            min_e_ea = np.abs(saliency - label).astype(np.float32)
        
            min_e_ea[label * (ea < e) == 1] = ea[label * (ea < e) == 1]
        
            b = np.ones_like(label).astype(np.float32)
            b[label == 0] = 2 - 1 * np.exp(np.log(1 - 0.5) / 5. * dst[label == 0])
        
            ew = min_e_ea * b
            tpw = np.sum(label) - np.sum(ew[label == 1])
            fpw = np.sum(ew[label == 0])
        
            rec = 1 - np.mean(ew[label == 1])  # Weighed Recall
            prec = tpw / (eps + tpw + fpw)  # Weighted Precision
    
        value = (1 + beta2) * (rec * prec) / (eps + (beta2 * prec) + rec)
        return value
    
    def evaluate_detection_results(self,results_folder, label_folder, bb_scale=1.0, file_list=None, bbs_size_range=None, save_pr_file=None):
        self.get_total_components(label_folder, file_list)
        result_ext = util.get_files_extension(results_folder)
        label_ext = util.get_files_extension(label_folder)
        for t in self.iou_thresholds:
            self.correct_comp_found[str(t)]=0.0
            self.wrong_comp_found[str(t)]=0.0
            self.missed_comp[str(t)]=0.0
            self.cumulative_prec[str(t)] = []
            self.cumulative_rec[str(t)] = []
        
        for i,filename in enumerate(file_list):
            if(len(filename.split(".")) > 1):
                filename=filename.split(".")[0]
            
            label_bbs, label_shape = util.load_bb(label_folder+"/"+filename,label_ext)
            bbs, image_shape = util.load_bb(results_folder+"/"+filename,result_ext, scale=bb_scale, bbs_size_range=bbs_size_range)
            if(len(bbs) > 0 and len(label_bbs) > 0):
                for t in self.iou_thresholds:
                    correct, missed, wrong = FLIMMetrics.compare_components_detection(label_bbs, bbs, iou_threshold=t)
                    #To visualize the drawn bounding boxes
                    #drawn_bb = util.draw_bbs(io.imread("/data/datasets/parasites/schisto-separated/orig/"+filename+".png"), bbs, label_bbs, 3)
                    #io.imsave("./bbs/"+filename+".png", drawn_bb)
                    self.correct_comp_found[str(t)]+=correct
                    self.wrong_comp_found[str(t)]+=wrong
                    self.missed_comp[str(t)]+=missed
                    tp = self.correct_comp_found[str(t)]
                    fp = self.wrong_comp_found[str(t)]
                    fn = self.missed_comp[str(t)]
                    
                    if(tp+fp > 0.0):
                        pre = tp/(tp+fp)
                    elif(fp > 0.0):
                        pre = 0.0
                    else:
                        pre = 1.0
                    if(tp + fn > 0.0):
                        rec = tp/(tp+fn)
                    else:
                        rec = 1.0
                    self.cumulative_prec[str(t)].append(pre)
                    self.cumulative_rec[str(t)].append(rec)
            elif(len(label_bbs) > 0):
                for t in self.iou_thresholds:
                    correct = 0
                    missed = len(label_bbs)
                    wrong = 0
                    self.correct_comp_found[str(t)]+=correct
                    self.wrong_comp_found[str(t)]+=wrong
                    self.missed_comp[str(t)]+=missed
                    tp = self.correct_comp_found[str(t)]
                    fp = self.wrong_comp_found[str(t)]
                    fn = self.missed_comp[str(t)]
                    if(tp+fp > 0.0):
                        pre = tp/(tp+fp)
                    elif(fp > 0.0):
                        pre = 0.0
                    else:
                        pre = 1.0
                    if(tp + fn > 0.0):
                        rec = tp/(tp+fn)
                    else:
                        rec = 1.0
                    self.cumulative_prec[str(t)].append(pre)
                    self.cumulative_rec[str(t)].append(rec)
            else:
                for t in self.iou_thresholds:
                    correct = 0
                    missed = 0
                    wrong = len(bbs)
                    self.correct_comp_found[str(t)]+=correct
                    self.wrong_comp_found[str(t)]+=wrong
                    self.missed_comp[str(t)]+=missed
                    tp = self.correct_comp_found[str(t)]
                    fp = self.wrong_comp_found[str(t)]
                    fn = self.missed_comp[str(t)]
                    if(tp+fp > 0.0):
                        pre = tp/(tp+fp)
                    elif(fp > 0.0):
                        pre = 0.0
                    else:
                        pre = 1.0
                    if(tp + fn > 0.0):
                        rec = tp/(tp+fn)
                    else:
                        rec = 1.0
                    self.cumulative_prec[str(t)].append(pre)
                    self.cumulative_rec[str(t)].append(rec)
                
        self.get_fscores()
        self.get_pr_curve()
        self.get_aps()
        self.computed_detection_metrics = True
        
    def copy_sal_metrics(self, metrics_with_sal, metrics_without_sal):
        metrics_without_sal.mae = metrics_with_sal.mae
        metrics_without_sal.wfscore = metrics_with_sal.wfscore
        if(self.use_smeasure):
            metrics_without_sal.smeasure = metrics_with_sal.smeasure
        metrics_without_sal.dices = metrics_with_sal.dices
        metrics_without_sal.emeasure = metrics_with_sal.emeasure
        metrics_without_sal.computed_saliency_metrics = True
    
    def get_sal_pr_curve(self, results_folder, label_folder, file_list=None):
        result_ext = "."+util.get_files_extension(results_folder)
        label_ext = "."+util.get_files_extension(label_folder)
                
        precisions = []
        recalls = []
        for t in range(0,256,5):
            precision = []
            recall = []
            for i,filename in enumerate(file_list):
                if(len(filename.split(".")) > 1):
                    filename=filename.split(".")[0]
    
                if os.path.isfile(label_folder+filename+label_ext):
                    label = (io.imread(label_folder+filename+label_ext)).astype(np.uint8)
                    label[label>0] = 1
                if os.path.isfile(results_folder+filename+result_ext):
                    saliency = io.imread(results_folder+filename+result_ext)
                if(len(saliency.shape) == 3):
                    saliency = saliency[:,:,0]

                if(label.sum() > 0):
                    bin_sal = np.zeros(saliency.shape)
                    bin_sal[saliency>=t] = 1
                    tp = (bin_sal[label>0]).sum()
                    pre = tp / (bin_sal.sum() + 1e-20)
                    rec = tp / (label.sum() + 1e-20)
                    precision.append(pre)
                    recall.append(rec)
            precisions.append(np.array(precision).mean())
            recalls.append(np.array(recall).mean())

        precisions = np.sort(np.array(precisions))
        recalls = np.flip(np.sort(np.array(recalls)))
        self.pr_curve = [precisions, recalls]

    def get_metrics_summary_all_val(self):
        return self.dices.mean()

    #The code for computing te saliency metrics was largely taken from https://github.com/Mehrdad-Noori/Saliency-Evaluation-Toolbox
    def evaluate_saliency_results(self,results_folder, label_folder, file_list=None, save_pr_file=None):
        if(file_list == None):
            files_list =  os.listdir(folder)
            
        result_ext = "."+util.get_files_extension(results_folder)
        label_ext = "."+util.get_files_extension(label_folder)
        self.wfscore = []
        self.mae = []
        if(self.use_smeasure):
            self.smeasure = []
        self.dices = []
        self.emeasure = []
        for i,filename in enumerate(file_list):
            if(len(filename.split(".")) > 1):
                filename=filename.split(".")[0]

            if os.path.isfile(results_folder+filename+result_ext):
                saliency = io.imread(results_folder+filename+result_ext)
            if(len(saliency.shape) == 3):
                saliency = saliency[:,:,0]
            bin_sal = util.binarize_saliency(saliency)
            if os.path.isfile(label_folder+filename+label_ext):
                label = (io.imread(label_folder+filename+label_ext)).astype(np.uint8)
                label[label>0] = 1

            #if(label.ndim > 2):
            #    label = label[:,:,0]
            flatten_sal = bin_sal.flatten()
            flatten_label = label.flatten()
            #dices.append(1 - dice(flatten_sal, flatten_label))
            self.mae.append(mean_absolute_error(flatten_sal, flatten_label))
            self.wfscore.append(FLIMMetrics.weighted_fmeasure(saliency, label)) 
            if(self.use_smeasure):
                self.smeasure.append(FLIMMetrics.s_measure(saliency, label))
            self.dices.append(FLIMMetrics.dice_score(label, bin_sal))
            self.emeasure.append(FLIMMetrics.e_measure(saliency, label))
        self.mae = np.array(self.mae)
        #self.get_sal_pr_curve(results_folder, label_folder, file_list)
        self.wfscore = np.array(self.wfscore)
        if(self.use_smeasure):
            self.smeasure = np.array(self.smeasure)
        self.dices = np.array(self.dices)
        self.emeasure = np.array(self.emeasure)
        self.computed_saliency_metrics = True


