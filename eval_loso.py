import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.utils import get_locations
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import (CCNN, GRU, LSTM, Conformer, EEGNet,
                             EEGNetAttentionDANN, VanillaTransformer)
from torcheeg.trainers.domain_adaption.dann_like import GradientReverse


class DeepConvNet(nn.Module):
    def __init__(self, num_electrodes=32, chunk_size=128, num_classes=2,
                 dropout=0.5):
        super(DeepConvNet, self).__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 5)),
                nn.BatchNorm2d(out_channels), nn.ELU(),
                nn.AvgPool2d((1, 2)), nn.Dropout(dropout))

        self.features = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5)),
            nn.Conv2d(25, 25, (num_electrodes, 1)),
            nn.BatchNorm2d(25), nn.ELU(), nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout), block(25, 50), block(50, 100),
            block(100, 200))
        with torch.no_grad():
            size = self.features(
                torch.zeros(1, 1, num_electrodes, chunk_size))
            size = size.flatten(1).shape[1]
        self.classifier = nn.Linear(size, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


MODEL_SPECS = {
    'eegnet': ('waveform', lambda: EEGNet(chunk_size=128,
                                          num_electrodes=32,
                                          num_classes=2)),
    'deepconvnet': ('waveform', lambda: DeepConvNet(chunk_size=128,
                                                    num_electrodes=32,
                                                    num_classes=2)),
    'conformer': ('waveform', lambda: Conformer(num_electrodes=32,
                                                sampling_rate=128,
                                                num_classes=2)),
    'proposed': ('waveform', lambda: EEGNetAttentionDANN(chunk_size=128,
                                                         num_electrodes=32,
                                                         num_classes=2)),
    'eegnet_attention': (
        'waveform',
        lambda: EEGNetAttentionDANN(chunk_size=128,
                                    num_electrodes=32,
                                    use_attention=True,
                                    use_dann=False,
                                    num_classes=2)),
    'eegnet_dann': (
        'waveform',
        lambda: EEGNetAttentionDANN(chunk_size=128,
                                    num_electrodes=32,
                                    use_attention=False,
                                    use_dann=True,
                                    num_classes=2)),
    'gru': ('waveform', lambda: GRU(num_electrodes=32,
                                    hid_channels=64,
                                    num_classes=2)),
    'lstm': ('waveform', lambda: LSTM(num_electrodes=32,
                                      hid_channels=64,
                                      num_classes=2)),
    'vanilla_transformer': (
        'waveform',
        lambda: VanillaTransformer(chunk_size=128,
                                   t_patch_size=32,
                                   num_electrodes=32,
                                   hid_channels=32,
                                   depth=3,
                                   heads=4,
                                   head_channels=8,
                                   mlp_channels=64,
                                   num_classes=2)),
    'ccnn': ('features', lambda: CCNN(in_channels=4,
                                      grid_size=(9, 9),
                                      num_classes=2)),
}


SEQUENCE_MODELS = {'gru', 'lstm', 'vanilla_transformer'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Leave-one-subject-out accuracy on DEAP.')
    parser.add_argument('--data-path',
                        default='./data_preprocessed_python',
                        help='Directory containing s01.dat ... s32.dat.')
    parser.add_argument('--cache-path',
                        default='./loso_cache',
                        help='Directory for dataset IO and split files.')
    parser.add_argument('--models',
                        nargs='+',
                        default=list(MODEL_SPECS),
                        choices=list(MODEL_SPECS),
                        help='Models to evaluate.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--weight-domain',
                        type=float,
                        default=1.0,
                        help='Weight of the adversarial domain loss. The '
                        'held-out subject is used as unlabeled target data.')
    parser.add_argument('--folds',
                        type=int,
                        default=None,
                        help='Limit the number of subjects for smoke tests.')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output',
                        default='./loso_results.csv',
                        help='CSV path for per-subject accuracies.')
    return parser.parse_args()


def build_dataset(family, args):
    common = {
        'root_path': args.data_path,
        'chunk_size': 128,
        'num_baseline': 1,
        'baseline_chunk_size': 128,
        'label_transform': transforms.Compose([
            transforms.Select('valence'),
            transforms.Binary(5.0)
        ]),
        'num_worker': args.num_workers,
        'verbose': False,
    }
    if family == 'waveform':
        return DEAPDataset(
            io_path=str(Path(args.cache_path) / 'waveform'),
            online_transform=transforms.Compose(
                [transforms.To2d(), transforms.ToTensor()]),
            **common)
    locations = get_locations(DEAPDataset.channels, DEAPDataset.grid)
    return DEAPDataset(
        io_path=str(Path(args.cache_path) / 'features'),
        offline_transform=transforms.Compose([
            transforms.BandDifferentialEntropy(apply_to_baseline=True),
            transforms.ToGrid(locations, apply_to_baseline=True)
        ]),
        online_transform=transforms.Compose(
            [transforms.BaselineRemoval(), transforms.ToTensor()]),
        **common)


def prepare_batch(batch, model_name, device):
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.reshape(-1).long().to(device, non_blocking=True)
    if model_name in SEQUENCE_MODELS:
        x = x.squeeze(1)
    return x, y


def run_epoch(model, loader, model_name, device, criterion, optimizer=None,
              collect=False):
    training = optimizer is not None
    model.train(training)
    correct = 0
    total = 0
    loss_sum = 0.0
    probabilities = []
    labels = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            x, y = prepare_batch(batch, model_name, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.shape[0]
            loss_sum += loss.item() * y.shape[0]
            if collect:
                probabilities.append(
                    torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
                labels.append(y.cpu().numpy())
    predictions = None
    if collect:
        predictions = (np.concatenate(probabilities), np.concatenate(labels))
    return loss_sum / total, correct / total, predictions


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def run_dann_epoch(model, loader, target_loader, model_name, device, criterion,
                   optimizer, epoch, total_epochs, weight_domain):
    model.train()
    correct = 0
    total = 0
    loss_sum = 0.0
    target_batches = cycle(target_loader)
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        x, y = prepare_batch(batch, model_name, device)
        x_target, _ = prepare_batch(next(target_batches), model_name, device)
        progress = (epoch + batch_idx / num_batches) / total_epochs
        alpha = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0

        source_features = model.extract_features(x)
        target_features = model.extract_features(x_target)
        logits = model.classifier(source_features)
        task_loss = criterion(logits, y)

        source_domain = model.predict_domain(
            GradientReverse.apply(source_features, alpha))
        target_domain = model.predict_domain(
            GradientReverse.apply(target_features, alpha))
        domain_loss = criterion(
            source_domain,
            torch.zeros(len(source_domain), dtype=torch.long, device=device))
        domain_loss = domain_loss + criterion(
            target_domain,
            torch.ones(len(target_domain), dtype=torch.long, device=device))
        loss = task_loss + weight_domain * domain_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.shape[0]
        loss_sum += loss.item() * y.shape[0]
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.cache_path).mkdir(parents=True, exist_ok=True)

    datasets = {}
    splitters = {}
    rows = []
    device = torch.device(args.device)

    for model_name in args.models:
        family, factory = MODEL_SPECS[model_name]
        if family not in datasets:
            datasets[family] = build_dataset(family, args)
            splitters[family] = LeaveOneSubjectOut(
                split_path=str(Path(args.cache_path) / f'split_{family}'))
        dataset = datasets[family]
        splitter = splitters[family]
        splitter._prepare(dataset, ['subject_id'])
        subjects = splitter.subjects
        if args.folds is not None:
            subjects = subjects[:args.folds]

        for fold, subject in enumerate(subjects):
            torch.manual_seed(args.seed + fold)
            train_dataset, test_dataset = next(
                splitter.split(dataset, subject=subject))
            train_loader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers)
            test_loader = DataLoader(test_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers)
            model = factory().to(device)
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()
            use_dann = getattr(model, 'domain_classifier', None) is not None
            for epoch in range(args.epochs):
                if use_dann:
                    run_dann_epoch(model, train_loader, test_loader,
                                   model_name, device, criterion, optimizer,
                                   epoch, args.epochs, args.weight_domain)
                else:
                    run_epoch(model, train_loader, model_name, device,
                              criterion, optimizer)
            _, accuracy, (probabilities, targets) = run_epoch(
                model, test_loader, model_name, device, criterion,
                collect=True)
            metrics = {
                'accuracy': accuracy_score(targets, probabilities > 0.5),
                'precision': precision_score(targets,
                                             probabilities > 0.5,
                                             zero_division=0),
                'recall': recall_score(targets,
                                       probabilities > 0.5,
                                       zero_division=0),
                'f1': f1_score(targets,
                               probabilities > 0.5,
                               zero_division=0),
                'roc_auc': roc_auc_score(targets, probabilities)
                if len(np.unique(targets)) == 2 else float('nan'),
            }
            rows.append({
                'model': model_name,
                'subject': subject,
                **metrics,
            })
            summary = ' '.join(f'{key}={value:.4f}'
                               for key, value in metrics.items())
            print(f'{model_name:20s} {subject:8s} {summary}', flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['model', 'subject', *metric_names])
        writer.writeheader()
        writer.writerows(rows)

    print('\nLOSO results (mean +/- std)')
    header = f'{"model":20s} ' + ' '.join(f'{name:>18s}'
                                           for name in metric_names)
    print(header)
    for model_name in args.models:
        model_rows = [row for row in rows if row['model'] == model_name]
        cells = []
        for name in metric_names:
            values = np.array([row[name] for row in model_rows],
                              dtype=float)
            cells.append(f'{np.nanmean(values):.4f}+/-{np.nanstd(values):.4f}')
        print(f'{model_name:20s} ' + ' '.join(f'{cell:>18s}'
                                               for cell in cells))


if __name__ == '__main__':
    main()
