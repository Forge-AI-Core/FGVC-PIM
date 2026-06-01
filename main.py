import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import wandb
import warnings
import matplotlib.pyplot as plt

from models.builder import MODEL_GETTER
from data.dataset import build_loader
from utils.costom_logger import timeLogger
from utils.config_utils import load_yaml, build_record_folder, get_args
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
from eval import evaluate, cal_train_metrics
from sklearn.metrics import precision_recall_fscore_support
from utils.device_utils import get_device

warnings.simplefilter("ignore")

def eval_freq_schedule(args, epoch: int):
    if epoch >= args.max_epochs * 0.95:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.9:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.8:
        args.eval_freq = 2

def set_environment(args, tlogger):
    
    print("Setting Environment...")

    # Assign optimal device globally (supports CUDA, MPS, CPU)
    args.device = get_device()
    
    ### = = = =  Dataset and Data Loader = = = =  
    tlogger.print("Building Dataloader....")
    
    train_loader, val_loader = build_loader(args)
    
    if train_loader is None and val_loader is None:
        raise ValueError("Find nothing to train or evaluate.")

    if train_loader is not None:
        print("    Train Samples: {} (batch: {})".format(len(train_loader.dataset), len(train_loader)))
    else:
        # raise ValueError("Build train loader fail, please provide legal path.")
        print("    Train Samples: 0 ~~~~~> [Only Evaluation]")
    if val_loader is not None:
        print("    Validation Samples: {} (batch: {})".format(len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")
    
    print(f"    Using device: {args.device}")

    tlogger.print()

    ### = = = =  Model = = = =  
    tlogger.print("Building Model....")
    model = MODEL_GETTER[args.model_name](
        use_fpn = args.use_fpn,
        fpn_size = args.fpn_size,
        use_selection = args.use_selection,
        num_classes = args.num_classes,
        num_selects = args.num_selects,
        use_combiner = args.use_combiner,
    ) # about return_nodes, we use our default setting
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # model = torch.nn.DataParallel(model, device_ids=None) # device_ids : None --> use all gpus.
    model.to(args.device)
    tlogger.print()
    
    """
    if you have multi-gpu device, you can use torch.nn.DataParallel in single-machine multi-GPU 
    situation and use torch.nn.parallel.DistributedDataParallel to use multi-process parallelism.
    more detail: https://pytorch.org/tutorials/beginner/dist_overview.html
    """
    
    if train_loader is None:
        return train_loader, val_loader, model, None, None, None, None, start_epoch
    
    ### = = = =  Optimizer = = = =  
    tlogger.print("Building Optimizer....")
    # Split params: backbone vs head (fpn, selector, combiner)
    backbone_params = list(model.backbone.parameters()) if hasattr(model, 'backbone') else []
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    head_lr = getattr(args, 'head_lr', args.max_lr)
    head_scale = head_lr / args.max_lr  # relative to backbone schedule

    param_groups = [
        {"params": backbone_params, "lr_scale": 1.0},
        {"params": head_params, "lr_scale": head_scale},
    ]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(param_groups, lr=args.max_lr, nesterov=True, momentum=0.9, weight_decay=args.wdecay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(param_groups, lr=args.max_lr, weight_decay=args.wdecay)

    if args.pretrained is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    tlogger.print()

    schedule = cosine_decay(args, len(train_loader))

    if args.use_amp and args.device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext
        if args.use_amp:
            print("    [Warning] AMP is only supported on CUDA. Disabling AMP.")
            args.use_amp = False

    return train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch


def train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader):

    optimizer.zero_grad()
    total_batchs = len(train_loader) # just for log
    show_progress = [x/10 for x in range(11)] # just for log
    progress_i = 0
    all_train_preds = []
    all_train_labels = []
    for batch_id, (ids, datas, labels) in enumerate(train_loader):
        model.train()
        """ = = = = adjust learning rate = = = = """
        iterations = epoch * len(train_loader) + batch_id
        adjust_lr(iterations, optimizer, schedule)

        batch_size = labels.size(0)

        """ = = = = forward and calculate loss = = = = """
        datas, labels = datas.to(args.device), labels.to(args.device)

        with amp_context():
            """
            [Model Return]
                FPN + Selector + Combiner --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1', 'comb_outs'
                FPN + Selector --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1'
                FPN --> return 'layer1', 'layer2', 'layer3', 'layer4' (depend on your setting)
                ~ --> return 'ori_out'
            
            [Retuen Tensor]
                'preds_0': logit has not been selected by Selector.
                'preds_1': logit has been selected by Selector.
                'comb_outs': The prediction of combiner.
            """
            outs = model(datas)

            loss = 0.
            for name in outs:
                
                if "select_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")
                    if args.lambda_s != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss(label_smoothing=getattr(args, 'label_smoothing', 0.0))(logit, 
                                                       labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += args.lambda_s * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")

                    if args.lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([batch_size * S, args.num_classes]) - 1
                        labels_0 = labels_0.to(args.device)
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += args.lambda_n * loss_n
                    else:
                        loss_n = 0.0

                elif "layer" in name:
                    if not args.use_fpn:
                        raise ValueError("FPN not use here.")
                    if args.lambda_b != 0:
                        ### here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = nn.CrossEntropyLoss(label_smoothing=getattr(args, 'label_smoothing', 0.0))(outs[name].mean(1), labels)
                        loss += args.lambda_b * loss_b
                    else:
                        loss_b = 0.0
                
                elif "comb_outs" in name:
                    if not args.use_combiner:
                        raise ValueError("Combiner not use here.")

                    if args.lambda_c != 0:
                        loss_c = nn.CrossEntropyLoss(label_smoothing=getattr(args, 'label_smoothing', 0.0))(outs[name], labels)
                        loss += args.lambda_c * loss_c
                    # combiner 기준 예측값 누적 (Precision/Recall/F1용)
                    with torch.no_grad():
                        preds = torch.argmax(outs[name], dim=1).cpu().tolist()
                        all_train_preds.extend(preds)
                        all_train_labels.extend(labels.cpu().tolist())

                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels, label_smoothing=getattr(args, 'label_smoothing', 0.0))
                    loss += loss_ori
            
            loss /= args.update_freq
        
        """ = = = = calculate gradient = = = = """
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        """ = = = = update model = = = = """
        if (batch_id + 1) % args.update_freq == 0:
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update() # next batch
            else:
                optimizer.step()
            optimizer.zero_grad()

        """ log (MISC) """
        if args.use_wandb and ((batch_id + 1) % args.log_freq == 0):
            model.eval()
            msg = {}
            msg['info/epoch'] = epoch + 1
            msg['info/lr'] = get_lr(optimizer)
            cal_train_metrics(args, msg, outs, labels, batch_size)
            wandb.log(msg)

        train_progress = (batch_id + 1) / total_batchs
        # print(train_progress, show_progress[progress_i])
        if train_progress > show_progress[progress_i]:
            print(".."+str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1

    return all_train_preds, all_train_labels


def main(args, tlogger):
    """
    save model last.pt and best.pt
    """

    train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch = set_environment(args, tlogger)

    best_acc = 0.0
    best_eval_name = "null"

    # 그래프용 지표 누적
    train_history = {"epoch": [], "acc": [], "precision": [], "recall": [], "f1": []}
    eval_history  = {"epoch": [], "acc": [], "precision": [], "recall": [], "f1": []}

    if args.use_wandb:
        wandb.init(entity=args.wandb_entity,
                   project=args.project_name,
                   name=args.exp_name,
                   config=args)
        wandb.run.summary["best_acc"] = best_acc
        wandb.run.summary["best_eval_name"] = best_eval_name
        wandb.run.summary["best_epoch"] = 0

    for epoch in range(start_epoch, args.max_epochs):

        """
        Train
        """
        if train_loader is not None:
            tlogger.print("Start Training {} Epoch".format(epoch+1))
            all_train_preds, all_train_labels = train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader)
            # Train 에포크 끝 - combiner 기준 Precision/Recall/F1 계산
            if len(all_train_preds) > 0:
                train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
                    all_train_labels, all_train_preds, average='macro', zero_division=0)
                train_combiner_acc = round(
                    100 * sum(p == l for p, l in zip(all_train_preds, all_train_labels)) / len(all_train_labels), 3)
                tlogger.print("....Train | ACC: {}% | Precision: {}% | Recall: {}% | F1-Score: {}%".format(
                    train_combiner_acc,
                    round(train_prec * 100, 3),
                    round(train_rec * 100, 3),
                    round(train_f1 * 100, 3)))
                train_history["epoch"].append(epoch + 1)
                train_history["acc"].append(train_combiner_acc)
                train_history["precision"].append(round(train_prec * 100, 3))
                train_history["recall"].append(round(train_rec * 100, 3))
                train_history["f1"].append(round(train_f1 * 100, 3))
            tlogger.print()
        else:
            from eval import eval_and_save
            eval_and_save(args, model, val_loader, tlogger)
            break

        eval_freq_schedule(args, epoch)

        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {"model": model_to_save.state_dict(), "optimizer": optimizer.state_dict(), "epoch":epoch}
        torch.save(checkpoint, args.save_dir + "backup/last.pt")

        if (epoch + 1) % args.eval_freq == 0:
            """
            Evaluation
            """
            acc = -1
            if val_loader is not None:
                tlogger.print("Start Evaluating {} Epoch".format(epoch + 1))
                acc, eval_name, accs = evaluate(args, model, val_loader)
                prec = accs.get("Precision", 0)
                rec = accs.get("Recall", 0)
                f1 = accs.get("F1-Score", 0)
                combiner_acc = accs.get("combiner-top-1", acc)
                tlogger.print("....Eval | ACC: {}% ({}%) | Precision: {}% | Recall: {}% | F1-Score: {}%".format(
                    max(combiner_acc, best_acc), combiner_acc, prec, rec, f1))
                tlogger.print()
                eval_history["epoch"].append(epoch + 1)
                eval_history["acc"].append(combiner_acc)
                eval_history["precision"].append(prec)
                eval_history["recall"].append(rec)
                eval_history["f1"].append(f1)

            if args.use_wandb:
                wandb.log(accs)

            if combiner_acc > best_acc:
                best_acc = combiner_acc
                best_eval_name = "combiner-top-1"
                torch.save(checkpoint, args.save_dir + "backup/best.pt")
            if args.use_wandb:
                wandb.run.summary["best_acc"] = best_acc
                wandb.run.summary["best_eval_name"] = best_eval_name
                wandb.run.summary["best_epoch"] = epoch + 1

    save_metrics_plots(args, train_history, eval_history)

    # Save final eval results with the best model
    if val_loader is not None and train_loader is not None:
        import os
        best_path = os.path.join(args.save_dir, "backup", "best.pt")
        if os.path.exists(best_path):
            tlogger.print("Loading best model for final evaluation...")
            checkpoint = torch.load(best_path, map_location=args.device, weights_only=False)
            model_to_load = model.module if hasattr(model, "module") else model
            model_to_load.load_state_dict(checkpoint['model'])
            from eval import eval_and_save
            eval_and_save(args, model, val_loader, tlogger)


def save_metrics_plots(args, train_history, eval_history):
    """ Train 그래프 """
    if len(train_history["epoch"]) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train_history["epoch"], train_history["acc"],       marker='o', label='ACC (combiner-top-1)', linewidth=2)
        ax.plot(train_history["epoch"], train_history["precision"], marker='s', label='Precision', linewidth=2, linestyle='--')
        ax.plot(train_history["epoch"], train_history["recall"],    marker='^', label='Recall', linewidth=2, linestyle='--')
        ax.plot(train_history["epoch"], train_history["f1"],        marker='D', label='F1-Score', linewidth=2, linestyle=':')
        ax.set_xlabel('Epoch', fontsize=13)
        ax.set_ylabel('Score (%)', fontsize=13)
        ax.set_title('Train Metrics - {}/{}'.format(args.project_name, args.exp_name), fontsize=14)
        ax.set_xticks(train_history["epoch"])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.save_dir + "train_metrics.png", dpi=150)
        plt.close()

    """ Eval 그래프 """
    if len(eval_history["epoch"]) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(eval_history["epoch"], eval_history["acc"],       marker='o', label='ACC (combiner-top-1)', linewidth=2)
        ax.plot(eval_history["epoch"], eval_history["precision"], marker='s', label='Precision', linewidth=2, linestyle='--')
        ax.plot(eval_history["epoch"], eval_history["recall"],    marker='^', label='Recall', linewidth=2, linestyle='--')
        ax.plot(eval_history["epoch"], eval_history["f1"],        marker='D', label='F1-Score', linewidth=2, linestyle=':')
        ax.set_xlabel('Epoch', fontsize=13)
        ax.set_ylabel('Score (%)', fontsize=13)
        ax.set_title('Eval Metrics - {}/{}'.format(args.project_name, args.exp_name), fontsize=14)
        ax.set_xticks(eval_history["epoch"])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.save_dir + "eval_metrics.png", dpi=150)
        plt.close()


if __name__ == "__main__":

    tlogger = timeLogger()

    tlogger.print("Reading Config...")
    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    build_record_folder(args)
    tlogger.print()

    main(args, tlogger)