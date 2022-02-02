from copy import deepcopy

import numpy as np
import torch


def make_val_from_train(dataset, split=.9):
    train_ds, val_ds = deepcopy(dataset), deepcopy(dataset)

    train_idx, val_idx = [], []
    for label in np.unique(dataset.targets):
        label_idx = np.squeeze(np.argwhere(dataset.targets == label))
        split_idx = int(label_idx.shape[0] * split)
        train_idx += [label_idx[:split_idx]]
        val_idx   += [label_idx[split_idx:]]

    train_idx = np.concatenate(train_idx)
    val_idx   = np.concatenate(val_idx)

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)

    return train_ds, val_ds


def get_data_and_tfs(args):
    dataset = {'cifar10' : CIFAR10,
               'cifar100': CIFAR100,
               'miniimagenet'  : MiniImagenet,
               'imagenet32' : Imagenet32}[args.dataset]

    if args.n_tasks == -1:
        args.n_tasks = dataset.default_n_tasks


    H = dataset.default_size
    args.input_size = (3, H, H)

    base_tf  = dataset.base_transforms()
    train_tf = dataset.train_transforms(use_augs=args.use_augs)
    eval_tf  = dataset.eval_transforms()

    ds_kwargs = {'root': args.data_root, 'download': args.download}

    val_ds = test_ds = None

    # if args.dataset == 'imagenet32':
    #     train_ds, _, test_ds = get_imagenet32(args) 
    if args.validation:
        trainval_ds      = dataset(train=True, **ds_kwargs)
        train_ds, val_ds = make_val_from_train(trainval_ds)
        train_ds.dataset.transform = base_tf
        val_ds.dataset.transform   = eval_tf
    else:
        train_ds         = dataset(train=True, transform=base_tf, **ds_kwargs)
        test_ds          = dataset(train=False, transform=eval_tf, **ds_kwargs)

    train_sampler = ContinualSampler(train_ds, args.n_tasks, args, smooth=args.smooth)
    train_loader  = torch.utils.data.DataLoader(
        train_ds,
        num_workers=args.num_workers,
        sampler=train_sampler,
        batch_size=args.batch_size,
        pin_memory=True
    )

    if val_ds is not None:
        val_sampler = ContinualSampler(val_ds, args.n_tasks, args)
        test_loader = None
        val_loader  = torch.utils.data.DataLoader(
            val_ds,
            num_workers=args.num_workers,
            batch_size=128,
            sampler=val_sampler,
            pin_memory=True
        )

    elif test_ds is not None:
        test_sampler  = ContinualSampler(test_ds,  args.n_tasks, args)
        val_loader  = None
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            num_workers=args.num_workers,
            batch_size=128,
            sampler=test_sampler,
            pin_memory=True
        )


    args.n_classes = train_sampler.n_classes
    args.n_classes_per_task = args.n_classes // args.n_tasks

    return train_tf, train_loader, val_loader, test_loader
