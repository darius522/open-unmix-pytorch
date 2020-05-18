import argparse
import model
import data
import torch
import time
from pathlib import Path
import tqdm
import json
import utils
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
import copy
import h5py


tqdm.monitor_interval = 0


def train(args, unmix, device, data_dict, train_sampler, optimizer):
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(range(args.num_it), disable=args.quiet)

    for n in pbar:

        x, y = data.SATBBatchGenerator(train_sampler,args,data_dict,partition='train')

        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        Y_hat = unmix(x)
        Y = unmix.transform(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
    return losses.avg

def valid(args, unmix, device, data_dict, valid_sampler):
    losses = utils.AverageMeter()
    unmix.eval()
    pbar = tqdm.tqdm(range(args.num_it), disable=args.quiet)

    for n in pbar:

        x, y = data.SATBBatchGenerator(valid_sampler,args,data_dict,partition='valid')
        pbar.set_description("Valid batch")
        x, y = x.to(device), y.to(device)
        Y_hat = unmix(x)
        Y = unmix.transform(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        losses.update(loss.item(), Y.size(1))
    return losses.avg

def createDictForDataset(dataset,partition):

    sources = ['soprano','tenor','bass','alto']
    allParts  = []
    partPerSongDict = {}

    # 1 Create dict of available parts per songs
    allParts = [key for key in dataset[partition].keys()]
    
    for part in allParts:
        songs = dataset[partition][part]
        for song in songs:
            if song in partPerSongDict:
                partPerSongDict[song].append(part)
            else:
                partPerSongDict[song] = [part]

    # List numbers of singer per part for a given song
    partCountPerSongDict = {}
    for song in partPerSongDict.keys():
        parts = partPerSongDict[song]
        parts = [x[:-1] for x in parts]
        partCountPerSongDict[song] = {i:parts.count(i) for i in parts}

        # If a part is missing from the song, add its key and 0 as part count
        diff = list(set(sources) - set(partCountPerSongDict[song].keys()))
        if len(diff) != 0:
            for missing_part in diff:
                partCountPerSongDict[song][missing_part] = 0

    return partCountPerSongDict

def get_statistics(args, dataset):
    scaler = sklearn.preprocessing.StandardScaler()

    spec = torch.nn.Sequential(
        model.STFT(n_fft=args.nfft, n_hop=args.nhop),
        model.Spectrogram(mono=True)
    )

    # dataset_scaler = copy.deepcopy(dataset)
    # dataset_scaler.samples_per_track = 1
    # dataset_scaler.augmentations = None
    # dataset_scaler.random_chunks = False
    # dataset_scaler.random_track_mix = False
    # dataset_scaler.random_interferer_mix = False
    # dataset_scaler.seq_duration = None
    pbar = tqdm.tqdm(dataset['train'].keys(), disable=args.quiet)
    for part in pbar:
        for song in dataset['train'][part].keys():
            x = torch.tensor(np.asarray(dataset['train'][part][song]['raw_wav'])[np.newaxis,...])
            #import pdb; pdb.set_trace()
            pbar.set_description("Compute dataset statistics")
            X = spec(x[None, ...])
            scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(
        scaler.scale_,
        1e-4*np.max(scaler.scale_)
    )
    return scaler.mean_, std


def main():
    parser = argparse.ArgumentParser(description='Open Unmix Trainer')

    # which target do we want to train?
    parser.add_argument('--target', type=str, default='soprano',
                        help='target source (will be passed to the dataset)')

    # Dataset paramaters
    parser.add_argument('--dataset', type=str, default="satb",
                        choices=[
                            'musdb', 'aligned', 'sourcefolder',
                            'trackfolder_var', 'trackfolder_fix'
                        ],
                        help='Name of the dataset.')
    parser.add_argument('--root', type=str, default="./", help='root path of dataset')
    parser.add_argument('--train-path', type=str, default="../../data/satb_dst/train/raw_audio", help='train files')
    parser.add_argument('--valid-path', type=str, default="../../data/satb_dst/valid/raw_audio", help='root path of dataset')
    parser.add_argument('--output', type=str, default="open-unmix",
                        help='provide output path base folder name')
    parser.add_argument('--model', type=str, help='Path to checkpoint folder')

    # Trainig Parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-it', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--patience', type=int, default=140,
                        help='maximum number of epochs to train (default: 140)')
    parser.add_argument('--lr-decay-patience', type=int, default=80,
                        help='lr decay patience for plateau scheduler')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                        help='gamma of learning rate scheduler decay')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # Model Parameters
    parser.add_argument('--seq-dur', type=float, default=4.0,
                        help='Sequence duration in seconds'
                        'value of <=0.0 will use full/variable length')
    parser.add_argument('--unidirectional', action='store_true', default=False,
                        help='Use unidirectional LSTM instead of bidirectional')
    parser.add_argument('--nfft', type=int, default=4096,
                        help='STFT fft size and window size')
    parser.add_argument('--nhop', type=int, default=1024,
                        help='STFT hop size')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='hidden size parameter of dense bottleneck layers')
    parser.add_argument('--bandwidth', type=int, default=22050,
                        help='maximum model bandwidth in herz')
    parser.add_argument('--nb-channels', type=int, default=1,
                        help='set number of channels for model (1, 2)')
    parser.add_argument('--nb-workers', type=int, default=0,
                        help='Number of workers for dataloader.')

    # Misc Parameters
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='less verbose during training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    print("Using Torchaudio: ", utils._torchaudio_available())
    dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}

    # repo_dir = os.path.abspath(os.path.dirname(__file__))
    # repo = Repo(repo_dir)
    # commit = repo.head.commit.hexsha[:7]

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isfile(os.path.join(args.root,"satb.hdf5")):
        data.createSATBDataset(args)

    dataset   = h5py.File(os.path.join(args.root,"satb.hdf5"), "r")

    dst_dict_train = createDictForDataset(dataset,partition='train')
    dst_dict_valid = createDictForDataset(dataset,partition='valid')

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    #import pdb; pdb.set_trace()
    # train_sampler = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     **dataloader_kwargs
    # )
    # valid_sampler = torch.utils.data.DataLoader(
    #     valid_dataset, batch_size=1,
    #     **dataloader_kwargs
    # )

    
    if args.model:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, dataset)

    max_bin = utils.bandwidth_to_max_bin(
        args.bandwidth, args.nfft, args.bandwidth
    )

    unmix = model.OpenUnmix(
        input_mean=scaler_mean,
        input_scale=scaler_std,
        nb_channels=args.nb_channels,
        hidden_size=args.hidden_size,
        n_fft=args.nfft,
        n_hop=args.nhop,
        max_bin=max_bin,
        sample_rate=args.bandwidth
    ).to(device)

    optimizer = torch.optim.Adam(
        unmix.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10
    )

    es = utils.EarlyStopping(patience=args.patience)

    # if a model is specified: resume training
    if args.model:
        model_path = Path(args.model).expanduser()
        with open(Path(model_path, args.target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # train for another epochs_trained
        t = tqdm.trange(
            results['epochs_trained'],
            results['epochs_trained'] + args.epochs + 1,
            disable=args.quiet
        )
        train_losses = results['train_loss_history']
        valid_losses = results['valid_loss_history']
        train_times = results['train_time_history']
        best_epoch = results['best_epoch']
        es.best = results['best_loss']
        es.num_bad_epochs = results['num_bad_epochs']
    # else start from 0
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = train(args, unmix, device, dst_dict_train, dataset, optimizer)
        valid_loss = valid(args, unmix, device, dst_dict_valid, dataset)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': unmix.state_dict(),
                'best_loss': es.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=args.target
        )

        # save params
        params = {
            'epochs_trained': epoch,
            'args': vars(args),
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs,
            #'commit': commit
        }

        with open(Path(target_path,  args.target + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()
