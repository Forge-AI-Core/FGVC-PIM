import numpy as np
import torch
import torch.nn.functional as F
from typing import Union
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt


@torch.no_grad()
def cal_train_metrics(args, msg: dict, outs: dict, labels: torch.Tensor, batch_size: int):
    """
    only present top-1 training accuracy
    """

    total_loss = 0.0

    if args.use_fpn:
        for i in range(1, 5):
            acc = top_k_corrects(outs["layer"+str(i)].mean(1), labels, tops=[1])["top-1"] / batch_size
            acc = round(acc * 100, 2)
            msg["train_acc/layer{}_acc".format(i)] = acc
            loss = F.cross_entropy(outs["layer"+str(i)].mean(1), labels)
            msg["train_loss/layer{}_loss".format(i)] = loss.item()
            total_loss += loss.item()

    if args.use_selection:
        for name in outs:
            if "select_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_0, tops=[1])["top-1"] / (B*S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            labels_0 = torch.zeros([B * S, args.num_classes]) - 1
            labels_0 = labels_0.to(args.device)
            loss = F.mse_loss(F.tanh(logit), labels_0)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

        for name in outs:
            if "drop_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_1, tops=[1])["top-1"] / (B*S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            loss = F.cross_entropy(logit, labels_1)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

    if args.use_combiner:
        acc = top_k_corrects(outs['comb_outs'], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/combiner_acc"] = acc
        loss = F.cross_entropy(outs['comb_outs'], labels)
        msg["train_loss/combiner_loss"] = loss.item()
        total_loss += loss.item()

    if "ori_out" in outs:
        acc = top_k_corrects(outs["ori_out"], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/ori_acc"] = acc
        loss = F.cross_entropy(outs["ori_out"], labels)
        msg["train_loss/ori_loss"] = loss.item()
        total_loss += loss.item()

    msg["train_loss/total_loss"] = total_loss



@torch.no_grad()
def top_k_corrects(preds: torch.Tensor, labels: torch.Tensor, tops: list = [1, 3, 5]):
    """
    preds: [B, C] (C is num_classes)
    labels: [B, ]
    """
    if preds.device != torch.device('cpu'):
        preds = preds.cpu()
    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    tmp_cor = 0
    corrects = {"top-"+str(x):0 for x in tops}
    sorted_preds = torch.sort(preds, dim=-1, descending=True)[1]
    for i in range(tops[-1]):
        tmp_cor += sorted_preds[:, i].eq(labels).sum().item()
        # records
        if "top-"+str(i+1) in corrects:
            corrects["top-"+str(i+1)] = tmp_cor
    return corrects


@torch.no_grad()
def _cal_evalute_metric(corrects: dict, 
                        total_samples: dict,
                        logits: torch.Tensor, 
                        labels: torch.Tensor, 
                        this_name: str,
                        scores: Union[list, None] = None, 
                        score_names: Union[list, None] = None):
    
    tmp_score = torch.softmax(logits, dim=-1)
    tmp_corrects = top_k_corrects(tmp_score, labels, tops=[1, 3]) # return top-1, top-3, top-5 accuracy
    
    ### each layer's top-1, top-3 accuracy
    for name in tmp_corrects:
        eval_name = this_name + "-" + name
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        corrects[eval_name] += tmp_corrects[name]
        total_samples[eval_name] += labels.size(0)
    
    if scores is not None:
        scores.append(tmp_score)
    if score_names is not None:
        score_names.append(this_name)


@torch.no_grad()
def _average_top_k_result(corrects: dict, total_samples: dict, scores: list, labels: torch.Tensor, tops: list = [1, 2, 3, 4, 5]):
    """
    scores is a list contain:
    [
        tensor1, 
        tensor2,...
    ] tensor1 and tensor2 have same size [B, num_classes]
    """
    # initial
    for t in tops:
        eval_name = "highest-{}".format(t)
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        total_samples[eval_name] += labels.size(0)

    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    
    batch_size = labels.size(0)
    scores_t = torch.cat([s.unsqueeze(1) for s in scores], dim=1) # B, 5, C

    if scores_t.device != torch.device('cpu'):
        scores_t = scores_t.cpu()

    max_scores = torch.max(scores_t, dim=-1)[0]
    # sorted_ids = torch.sort(max_scores, dim=-1, descending=True)[1] # this id represents different layers outputs, not samples

    for b in range(batch_size):
        tmp_logit = None
        ids = torch.sort(max_scores[b], dim=-1)[1] # S
        for i in range(tops[-1]):
            top_i_id = ids[i]
            if tmp_logit is None:
                tmp_logit = scores_t[b][top_i_id]
            else:
                tmp_logit += scores_t[b][top_i_id]
            # record results
            if i+1 in tops:
                if torch.max(tmp_logit, dim=-1)[1] == labels[b]:
                    eval_name = "highest-{}".format(i+1)
                    corrects[eval_name] += 1


def evaluate(args, model, test_loader):
    """
    [Notice: Costom Model]
    If you use costom model, please change fpn module return name (under 
    if args.use_fpn: ...)
    [Evaluation Metrics]
    We calculate each layers accuracy, combiner accuracy and average-higest-1 ~ 
    average-higest-5 accuracy (average-higest-5 means average all predict scores
    as final predict)
    """

    model.eval()
    corrects = {}
    total_samples = {}
    all_preds = []
    all_labels = []

    total_batchs = len(test_loader) # just for log
    show_progress = [x/10 for x in range(11)] # just for log
    progress_i = 0

    with torch.no_grad():
        """ accumulate """
        for batch_id, (ids, datas, labels) in enumerate(test_loader):

            score_names = []
            scores = []
            datas = datas.to(args.device)

            outs = model(datas)

            if args.use_fpn:
                for i in range(1, 5):
                    this_name = "layer" + str(i)
                    _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name)

            if args.use_combiner:
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, outs["comb_outs"], labels, this_name)
                # combiner 기준 Precision/Recall/F1을 위해 누적
                preds = torch.argmax(outs["comb_outs"], dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

            if "ori_out" in outs:
                this_name = "original"
                _cal_evalute_metric(corrects, total_samples, outs["ori_out"], labels, this_name)

            eval_progress = (batch_id + 1) / total_batchs

            if eval_progress > show_progress[progress_i]:
                print(".."+str(int(show_progress[progress_i]*100))+"%", end='', flush=True)
                progress_i += 1

        """ calculate accuracy """
        best_top1 = 0.0
        best_top1_name = ""
        eval_acces = {}
        for name in corrects:
            acc = corrects[name] / total_samples[name]
            acc = round(100 * acc, 3)
            eval_acces[name] = acc
            ### only compare top-1 accuracy
            if "top-1" in name or "highest" in name:
                if acc >= best_top1:
                    best_top1 = acc
                    best_top1_name = name

        """ calculate Precision / Recall / F1 (combiner 기준) """
        if len(all_preds) > 0:
            prec, rec, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0)
            eval_acces["Precision"] = round(prec * 100, 3)
            eval_acces["Recall"] = round(rec * 100, 3)
            eval_acces["F1-Score"] = round(f1 * 100, 3)

            # Per-class metrics
            cm = confusion_matrix(all_labels, all_preds)
            class_precs, class_recs, class_f1s, _ = precision_recall_fscore_support(
                all_labels, all_preds, average=None, zero_division=0)

            try:
                dataset_root = test_loader.dataset.root
                class_names = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
                class_names.sort()
            except Exception:
                class_names = [f"class_{i}" for i in range(len(class_precs))]

            for i, name in enumerate(class_names):
                if i < len(class_precs):
                    row_sum = cm[i].sum()
                    class_acc = cm[i][i] / row_sum if row_sum > 0 else 0.0
                    eval_acces[f"class_{name}_ACC"] = round(class_acc * 100, 3)
                    eval_acces[f"class_{name}_Precision"] = round(class_precs[i] * 100, 3)
                    eval_acces[f"class_{name}_Recall"] = round(class_recs[i] * 100, 3)
                    eval_acces[f"class_{name}_F1-Score"] = round(class_f1s[i] * 100, 3)

    return best_top1, best_top1_name, eval_acces


def evaluate_cm(args, model, test_loader):
    """
    [Notice: Costom Model]
    If you use costom model, please change fpn module return name (under
    if args.use_fpn: ...)
    [Evaluation Metrics]
    We calculate each layers accuracy, combiner accuracy and average-higest-1 ~
    average-higest-5 accuracy (average-higest-5 means average all predict scores
    as final predict)
    """

    model.eval()
    corrects = {}
    total_samples = {}
    results = []

    with torch.no_grad():
        """ accumulate """
        for batch_id, (ids, datas, labels) in enumerate(test_loader):

            score_names = []
            scores = []
            datas = datas.to(args.device)
            outs = model(datas)

            # if args.use_fpn and (0 < args.highest < 5):
            #     this_name = "layer" + str(args.highest)
            #     _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)

            if args.use_combiner:
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, outs["comb_outs"], labels, this_name, scores, score_names)

            # _average_top_k_result(corrects, total_samples, scores, labels)

            for i in range(scores[0].shape[0]):
                results.append([test_loader.dataset.data_infos[ids[i].item()]['path'], int(labels[i].item()),
                                int(scores[0][i].argmax().item()),
                                scores[0][i][scores[0][i].argmax().item()].item()])  # 图片路径，标签，预测标签，得分

        """ wirte xlsx"""
        writer = pd.ExcelWriter(args.save_dir + 'infer_result.xlsx')
        df = pd.DataFrame(results, columns=["id", "original_label", "predict_label", "goal"])
        df.to_excel(writer, index=False, sheet_name="Sheet1")
        writer.close()

        """ calculate accuracy """

        best_top1 = 0.0
        best_top1_name = ""
        eval_acces = {}
        for name in corrects:
            acc = corrects[name] / total_samples[name]
            acc = round(100 * acc, 3)
            eval_acces[name] = acc
            ### only compare top-1 accuracy
            if "top-1" in name or "highest" in name:
                if acc > best_top1:
                    best_top1 = acc
                    best_top1_name = name

        """ wirte xlsx"""
        results_mat = np.asmatrix(results)
        y_actual = results_mat[:, 1].transpose().tolist()[0]
        y_actual = list(map(int, y_actual))
        y_predict = results_mat[:, 2].transpose().tolist()[0]
        y_predict = list(map(int, y_predict))

        if len(y_predict) > 0:
            # 1. Macro Averaged
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_actual, y_predict, average='macro', zero_division=0)
            eval_acces["Precision"] = round(prec * 100, 3)
            eval_acces["Recall"] = round(rec * 100, 3)
            eval_acces["F1-Score"] = round(f1 * 100, 3)

            # 2. Per-class metrics
            cm = confusion_matrix(y_actual, y_predict)
            class_precs, class_recs, class_f1s, _ = precision_recall_fscore_support(
                y_actual, y_predict, average=None, zero_division=0)

            try:
                dataset_root = test_loader.dataset.root
                class_names = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
                class_names.sort()
            except Exception:
                class_names = [f"class_{i}" for i in range(len(class_precs))]

            for i, name in enumerate(class_names):
                if i < len(class_precs):
                    row_sum = cm[i].sum()
                    class_acc = cm[i][i] / row_sum if row_sum > 0 else 0.0
                    eval_acces[f"class_{name}_ACC"] = round(class_acc * 100, 3)
                    eval_acces[f"class_{name}_Precision"] = round(class_precs[i] * 100, 3)
                    eval_acces[f"class_{name}_Recall"] = round(class_recs[i] * 100, 3)
                    eval_acces[f"class_{name}_F1-Score"] = round(class_f1s[i] * 100, 3)

        folders = [f for f in os.listdir(args.val_root) if os.path.isdir(os.path.join(args.val_root, f))]
        folders.sort()  # sort by alphabet
        print("[dataset] class:", folders)
        df_confusion = confusion_matrix(y_actual, y_predict)
        plot_confusion_matrix(df_confusion, folders, args.save_dir + "infer_cm.png", accuracy=best_top1)

    return best_top1, best_top1_name, eval_acces


@torch.no_grad()
def eval_and_save(args, model, val_loader, tlogger):
    tlogger.print("Start Evaluating")
    acc, eval_name, eval_acces = evaluate(args, model, val_loader)
    tlogger.print("....BEST_ACC: {} {}%".format(eval_name, acc))
    ### build records.txt
    msg = "[Evaluation Results]\n"
    msg += "Project: {}, Experiment: {}\n".format(args.project_name, args.exp_name)
    msg += "Samples: {}\n".format(len(val_loader.dataset))
    msg += "\n"
    
    # 1. 일반 레이어 성능 출력
    for name in eval_acces:
        if not name.startswith("class_") and name not in ["Precision", "Recall", "F1-Score"]:
            msg += "    {} {}%\n".format(name, eval_acces[name])
    msg += "\n"
    
    # 2. 글로벌 결합 성능 요약
    msg += "[Overall Performance]\n"
    msg += "    Precision: {}%\n".format(eval_acces.get("Precision", 0))
    msg += "    Recall: {}%\n".format(eval_acces.get("Recall", 0))
    msg += "    F1-Score: {}%\n".format(eval_acces.get("F1-Score", 0))
    msg += "\n"
    
    # 3. 클래스별 성능 요약
    class_keys = sorted([k for k in eval_acces.keys() if k.startswith("class_")])
    if len(class_keys) > 0:
        msg += "[Per-Class Performance (Combiner)]\n"
        classes_data = {}
        for k in class_keys:
            parts = k.split("_")
            metric = parts[-1]
            cname = "_".join(parts[1:-1])
            if cname not in classes_data:
                classes_data[cname] = {}
            classes_data[cname][metric] = eval_acces[k]
            
        for cname, metrics in classes_data.items():
            msg += "  - Class '{}':\n".format(cname)
            msg += "    ACC: {}% | Precision: {}% | Recall: {}% | F1-Score: {}%\n".format(
                metrics.get("ACC", 0), metrics.get("Precision", 0), metrics.get("Recall", 0), metrics.get("F1-Score", 0)
            )
        msg += "\n"
        
    msg += "BEST_ACC: {} {}% ".format(eval_name, acc)

    with open(args.save_dir + "eval_results.txt", "w") as ftxt:
        ftxt.write(msg)


@torch.no_grad()
def eval_and_cm(args, model, val_loader, tlogger):
    tlogger.print("Start Evaluating")
    acc, eval_name, eval_acces = evaluate_cm(args, model, val_loader)
    tlogger.print("....BEST_ACC: {} {}%".format(eval_name, acc))
    ### build records.txt
    msg = "[Evaluation Results]\n"
    msg += "Project: {}, Experiment: {}\n".format(args.project_name, args.exp_name)
    msg += "Samples: {}\n".format(len(val_loader.dataset))
    msg += "\n"
    
    # 1. 일반 레이어 성능 출력
    for name in eval_acces:
        if not name.startswith("class_") and name not in ["Precision", "Recall", "F1-Score"]:
            msg += "    {} {}%\n".format(name, eval_acces[name])
    msg += "\n"
    
    # 2. 글로벌 결합 성능 요약
    msg += "[Overall Performance]\n"
    msg += "    Precision: {}%\n".format(eval_acces.get("Precision", 0))
    msg += "    Recall: {}%\n".format(eval_acces.get("Recall", 0))
    msg += "    F1-Score: {}%\n".format(eval_acces.get("F1-Score", 0))
    msg += "\n"
    
    # 3. 클래스별 성능 요약
    class_keys = sorted([k for k in eval_acces.keys() if k.startswith("class_")])
    if len(class_keys) > 0:
        msg += "[Per-Class Performance (Combiner)]\n"
        classes_data = {}
        for k in class_keys:
            parts = k.split("_")
            metric = parts[-1]
            cname = "_".join(parts[1:-1])
            if cname not in classes_data:
                classes_data[cname] = {}
            classes_data[cname][metric] = eval_acces[k]
            
        for cname, metrics in classes_data.items():
            msg += "  - Class '{}':\n".format(cname)
            msg += "    ACC: {}% | Precision: {}% | Recall: {}% | F1-Score: {}%\n".format(
                metrics.get("ACC", 0), metrics.get("Precision", 0), metrics.get("Recall", 0), metrics.get("F1-Score", 0)
            )
        msg += "\n"
        
    msg += "BEST_ACC: {} {}% ".format(eval_name, acc)

    with open(args.save_dir + "infer_results.txt", "w") as ftxt:
        ftxt.write(msg)


def plot_confusion_matrix(cm, label_names, save_name, title='Confusion Matrix acc = ', accuracy=0):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(len(label_names) / 2, len(label_names) / 2), dpi=100)
    np.set_printoptions(precision=2)
    # print("cm:\n",cm)

    # 统计混淆矩阵中每格的概率值
    x, y = np.meshgrid(np.arange(len(cm)), np.arange(len(cm)))
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        try:
            c = (cm[y_val][x_val] / np.sum(cm, axis=1)[y_val]) * 100
        except KeyError:
            c = 0
        if c > 0.001:
            plt.text(x_val, y_val, "%0.1f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title + str('{:.3f}'.format(accuracy)))
    plt.colorbar()
    plt.xticks(np.arange(len(label_names)), label_names, rotation=45)
    plt.yticks(np.arange(len(label_names)), label_names)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(label_names))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(save_name, format='png')
    # plt.show()
