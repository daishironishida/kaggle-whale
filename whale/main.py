import argparse
from itertools import islice
import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn, cuda
from torch.optim import Adam, SGD, lr_scheduler
import tqdm

from . import models
from .dataset import TrainDataset, ValDataset, TestDataset, DATA_ROOT
from .transforms import get_transforms
from .utils import (
    write_event, load_model, ThreadingDataLoader as DataLoader, ON_KAGGLE)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict_valid', 'predict_test'])
    arg('run_root')
    arg('--model', default='resnet101')
    arg('--prev-model', default='none')
    arg('--pretrained', type=int, default=1)
    arg('--batch-size', type=int, default=64)
    arg('--step', type=int, default = 1)
    arg('--workers', type=int, default=2 if ON_KAGGLE else 4)
    arg('--lr', type=float, default=1e-4)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=100)
    arg('--epoch-size', type=int)
    arg('--debug', action='store_true')
    arg('--limit', type=int)
    arg('--loss', default='bce')
    arg('--transform', default='original')
    arg('--image-size', type=int, default=288)
    arg('--dropout', type=float, default=0)
    arg('--verbose', action='store_true')
    arg('--optim', default='adam')
    arg('--scheduler', default='none')
    arg('--schedule-length', type=int, default=0)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    train_root = DATA_ROOT / 'train'

    df_identified = pd.read_csv('df_identified.csv', index_col="Image")
    df_train = pd.read_csv('df_train.csv', index_col="Image")
    df_val = pd.read_csv('df_val.csv', index_col="Image")

    if args.limit:
        df_train = df_train[:args.limit]
        df_val = df_val[:args.limit]

    if args.loss == 'bce':
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        print('invalid loss function')
        return False

    train_transform, test_transform = get_transforms(args.transform, args.image_size)

    model = getattr(models, args.model)(
        num_features=512, pretrained=args.pretrained, dropout=args.dropout)
    use_cuda = cuda.is_available()
    fresh_params = list(model.fresh_params())
    all_params = list(model.parameters())
    if use_cuda:
        model = model.cuda()

    if args.prev_model != 'none' and not (run_root / 'model.pt').exists():
        shutil.copy(os.path.join(args.prev_model, 'best-model.pt'), str(run_root))
        shutil.copy(os.path.join(args.prev_model, 'model.pt'), str(run_root))

    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_loader = DataLoader(
            dataset=TrainDataset(train_root, df_train, train_transform, debug=args.debug),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        valid_loader = DataLoader(
            dataset=ValDataset(train_root, df_val, test_transform),
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        print(f'{len(train_loader.dataset):,} items in train, '
              f'{len(valid_loader.dataset):,} in valid')

        if args.optim == 'sgd':
            init_optimizer = SGD # no momentum
        elif args.optim == 'sgd_mom':
            init_optimizer = lambda params, lr: SGD(params, lr, momentum=0.9)
        else:
            init_optimizer = Adam

        train_kwargs = dict(
            args=args,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            patience=args.patience,
            init_optimizer=init_optimizer,
            use_cuda=use_cuda,
        )

        if args.pretrained and args.prev_model == 'none':
            if train(params=fresh_params, n_epochs=1, **train_kwargs):
                train(params=all_params, **train_kwargs)
        else:
            train(params=all_params, **train_kwargs)

    elif args.mode == 'validate':
        valid_loader = DataLoader(
            dataset=ValDataset(train_root, df_val, test_transform),
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        load_model(model, run_root / 'model.pt')
        validation(args, model, criterion,
                   tqdm.tqdm(valid_loader, desc='Validation', disable=not args.verbose),
                   use_cuda=use_cuda)

    elif args.mode.startswith('predict'):
        load_model(model, run_root / 'best-model.pt')
        predict_kwargs = dict(
            batch_size=args.batch_size,
            transform=test_transform,
            use_cuda=use_cuda,
            workers=args.workers,
        )
        train_loader = DataLoader(
            dataset=TestDataset(train_root, df_train, test_transform),
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )

        if args.mode == 'predict_valid':
            test_loader = DataLoader(
                dataset=TestDataset(train_root, df_val, test_transform),
                shuffle=False,
                batch_size=args.batch_size,
                num_workers=args.workers,
            )
            predict(args, model, train_loader, test_loader, df_identified,
                    out_path=run_root / 'val.h5', **predict_kwargs)

        elif args.mode == 'predict_test':
            test_root = DATA_ROOT / 'test'
            ss = pd.read_csv(DATA_ROOT / 'sample_submission.csv', index_col='Image')
            if args.limit:
                ss = ss[:args.limit]
            test_loader = DataLoader(
                dataset=TestDataset(test_root, ss, test_transform),
                shuffle=False,
                batch_size=args.batch_size,
                num_workers=args.workers,
            )
            predict(args, model, train_loader, test_loader, df_identified,
                    out_path=run_root / 'test.h5', **predict_kwargs)


def predict(args, model, train_loader, test_loader, df,
            out_path: Path, batch_size: int, transform: Callable,
            workers: int, use_cuda: bool):
    whale_names = sorted(set(df.Id))
    id_to_image = {}
    for name in whale_names:
        id_to_image[name] = list(df[df.Id == name].Image)
    
    model.eval()
    train_features, test_features = {}, {}
    with torch.no_grad():
        for inputs, image in tqdm.tqdm(train_loader, desc='Train Features', disable=not args.verbose):
            if use_cuda:
                inputs = inputs.cuda()
            outputs = model.forward_once(inputs)
            train_features[image] = outputs.data.cpu().numpy()
        for inputs, image in tqdm.tqdm(test_loader, desc='Test features', disable=not args.verbose):
            if use_cuda:
                inputs = inputs.cuda()
            outputs = model.forward_once(inputs)
            test_features[image] = outputs.data.cpu().numpy()

    test_preds = {}
    for test_img, test_feat in test_features.items():
        if use_cuda:
            test_feat = test_feat.cuda()
        preds = []
        for name in whale_names:
            best_score = float('int')
            for train_img in id_to_image[name]:
                train_feat = train_features[train_img]
                if use_cuda:
                    train_feat = train_feat.cuda()
                with torch.no_grad():
                    score = model.forward_head()
                score = float(score[:, 1])
                if score < best_score:
                    best_score = score
            preds.append(best_score)
        test_preds[test_img] = np.array(preds)

    pickle.dump(test_preds, out_path)
    print(f'Saved predictions to {out_path}')


def train(args, model: nn.Module, criterion, *, params,
          train_loader, valid_loader, init_optimizer, use_cuda,
          n_epochs=None, patience=2, max_lr_changes=2) -> bool:
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    params = list(params)
    optimizer = init_optimizer(params, lr)

    run_root = Path(args.run_root)
    model_path = run_root / 'model.pt'
    best_model_path = run_root / 'best-model.pt'
    if model_path.exists():
        state = load_model(model, model_path)
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')
    lr_changes = 0

    if args.scheduler != 'none':

        slope = 0.9 / args.schedule_length

        if args.scheduler == 'one_cycle':

            def lr_func(_):
                if step < args.schedule_length:
                    return slope * step + 0.1
                elif step <= args.schedule_length * 2:
                    return slope * (args.schedule_length - step) + 1
                else:
                    return 0.1 * slope * (2 * args.schedule_length - step) + 0.1

        elif args.scheduler == 'linear':

            step_first = step
            def lr_func(_):
                return -slope * (step - step_first) + 1

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
        lr = optimizer.param_groups[0]['lr']

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    report_each = 1000
    log = run_root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(
            total=(args.epoch_size or len(train_loader) * args.batch_size),
            disable=not args.verbose)
        tq.set_description(f'Epoch {epoch}, lr {lr:.3g}')
        losses, corrects, counts = [], [], []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0
            for i, (input0, input1, label) in enumerate(tl):
                if use_cuda:
                    input0, input1, label = input1.cuda(), input1.cuda(), label.cuda()
                output = model(input0, input1)
                _, preds = torch.max(output, 1)
                loss = _reduce_loss(criterion(output, label))
                batch_size = input0.size(0)
                (batch_size * loss).backward()
                if (i + 1) % args.step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                    if args.scheduler != 'none':
                        scheduler.step()
                        lr = optimizer.param_groups[0]['lr']
                tq.update(batch_size)
                losses.append(loss.item())
                corrects.append(torch.sum(preds == label.data).cpu().numpy())
                counts.append(batch_size)
                mean_loss = np.mean(losses[-report_each:])
                accuracy = np.sum(corrects[-report_each:])/np.sum(counts[-report_each:])
                tq.set_postfix(loss=f'{mean_loss:.3f}', acc=f'{accuracy:.3f}')
                if i and i % report_each == 0:
                    write_event(log, epoch, step, lr, loss=mean_loss, acc=accuracy)
            write_event(log, epoch, step, lr, loss=mean_loss, acc=accuracy)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(args, model, criterion, valid_loader, use_cuda)
            write_event(log, epoch, step, lr, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
            elif (patience and epoch - lr_reset_epoch > patience and
                  min(valid_losses[-patience:]) > best_valid_loss and
                  args.scheduler == 'patience'):
                # "patience" epochs without improvement
                lr_changes += 1
                if lr_changes > max_lr_changes:
                    break
                lr /= 5
                print(f'lr updated to {lr}')
                lr_reset_epoch = epoch
                optimizer = init_optimizer(params, lr)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return False
    return True


def validation(
        args, model: nn.Module, criterion, valid_loader, use_cuda,
        ) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for input0, input1, label in valid_loader:
            all_targets.append(label.numpy().copy())
            if use_cuda:
                input0, input1, label = input0.cuda(), input1.cuda(), label.cuda()
            output = model(input0, input1)
            _, preds = torch.max(output, 1)
            loss = criterion(output, label)
            all_losses.append(_reduce_loss(loss).item())
            all_predictions.append(preds.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    metrics = {}
    metrics['valid_acc'] = np.sum(all_predictions==all_targets) / len(all_predictions)
    metrics['valid_loss'] = np.mean(all_losses)
    if args.verbose:
        print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
            metrics.items(), key=lambda kv: -kv[1])))

    return metrics


def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]


if __name__ == '__main__':
    main()
