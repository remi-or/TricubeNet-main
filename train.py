from typing import List
import os, time
import numpy as np
import random
from argparse import Namespace

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from basenet.model import Model_factory
from loader import ListDataset
from loss import SWM_FPEM_Loss 
from utils.lr_scheduler import WarmupPolyLR
from utils.augmentations import Augmentation, Augmentation_test
    
cudnn.benchmark = True
PRINT_FREQ = 50

################################

def setup(
    root: str,
    classes: List[str],
    in_size: int,
    batch_size: int = 8,
    workers: int = 4,
    backbone: str = 'hourglass104_MRCB_cascade',
    epochs: int = 120,
    lr: float = 2.5e-4,
    alpha: float = 10.0,
    resume: str = '',
    seed: int = 1,
    save_path: str = './weight',
    amp: bool = True,
) -> Namespace:
    return Namespace(root=root, classes=classes, in_size=in_size,
                     batch_size=batch_size, workers=workers,
                     backbone=backbone, epochs=epochs, lr=lr,
                     alpha=alpha, resume=resume, seed=seed,
                     save_path=save_path, amp=amp, curr_iter=0)

def main(args: Namespace) -> None:
    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Set dimensions and pixel norm.
    args.in_size = (args.in_size, args.in_size) if isinstance(args.in_size, int) else args.in_size
    args.out_size = (args.in_size[0]//2, args.in_size[1]//2)
    mean = (0.485, 0.456, 0.406)
    var = (0.229, 0.224, 0.225)
    # Set the device
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    # Generate model, optimizer and loss
    num_classes = len(args.classes)
    model = Model_factory(args.backbone, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)  
    criterion = SWM_FPEM_Loss(num_classes=num_classes, alpha=args.alpha)
    # Resume if needed  
    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        state = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state['model'], strict=True)
        print("=> loaded checkpoint ", args.resume)
    elif args.resume:
        print(f"=> no checkpoint found at '{args.resume}'")
    # Generate datasets
    transform_train = Augmentation(args.in_size, mean, var)
    transform_val = Augmentation_test(args.in_size, mean, var)
    print('Using trainval split for training and validation.')
    train_dataset = ListDataset(args.root, args.classes, args.out_size, 'trainval', transform_train)
    val_dataset = ListDataset(args.root, args.classes, args.out_size, 'trainval', transform_val)
    print(f"Number of train = {len(train_dataset)} / valid = {len(val_dataset)}")
    # Generate dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, args.batch_size, True, None,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, args.batch_size, False, None,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    # Lr scheduling
    train_iter = len(train_loader) * args.epochs
    scheduler = WarmupPolyLR(
        optimizer, train_iter,
        warmup_iters=1000, power=0.90)
    # Training
    best_loss = 999999999
    best_dist = 999999999
    start = time.time()
    for epoch in range(0, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, device, start, epoch, args)
        # evaluate on validation set
        val_loss, val_dist = validate(val_loader, model, criterion, device, epoch, args)
        # save checkpoint
        if best_loss <= val_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, "best_loss", args.save_path)
        if best_dist <= val_dist:
            best_dist = val_dist
            save_checkpoint(model, optimizer, epoch, "best_dist", args.save_path)
                    
    
def train(train_loader, model, criterion, optimizer, scheduler, device, start, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    
    world_size = 1
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for x, y, w, s in train_loader:
        args.curr_iter += 1
        
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            outs = model(x)

            if type(outs) == list:
                loss = 0
                for out in outs:
                    loss += criterion(y, out, w, s)
                    
                loss /= len(outs)
                    
                outs = outs[-1]

            else:
                loss = criterion(y, outs, w, s)
        
        # compute gradient and backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        reduced_loss = reduce_tensor(loss.data, world_size)
        losses.update(reduced_loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        
        if args.curr_iter % PRINT_FREQ == 0:
            train_log = "Epoch: [%d/%d][%d/%d] " % (epoch, args.epochs, args.curr_iter, args.train_iter)
            train_log += "({0:.1f}%, {1:.1f} min) | ".format(args.curr_iter/args.train_iter*100, (end-start) / 60)
            train_log += "Time %.1f ms | Left %.1f min | " % (batch_time.avg * 1000, (args.train_iter - args.curr_iter) * batch_time.avg / 60)
            train_log += "Loss %.6f " % (losses.avg)
            print(train_log)

                
    
def validate(valid_loader, model, criterion, device, args):
    losses = AverageMeter()
    distances = AverageMeter()

    # switch to evaluate mode
    model.eval()

    world_size = 1
    end = time.time()

    for x, y, w, s in valid_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)

        # compute output
        with torch.no_grad():
            outs = model(x)
            
            if type(outs) == list:
                outs = outs[-1]

            loss = criterion(y, outs, w, s)

        # measure accuracy and record loss
        dist = torch.sqrt((y - outs)**2).mean()

        reduced_loss = reduce_tensor(loss.data, world_size)
        reduced_dist = reduce_tensor(dist.data, world_size)

        losses.update(reduced_loss.item())
        distances.update(reduced_dist.item())

    if args.local_rank == 0:
        valid_log = "\n============== validation ==============\n"
        valid_log += "valid time : %.1f s | " % (time.time() - end)
        valid_log += "valid loss : %.6f | " % (losses.avg)
        valid_log += "valid dist : %.6f \n" % (distances.avg)
        print(valid_log)
        
    return losses.avg, distances.avg


def save_checkpoint(model, optimizer, epoch, name, save_path):
    state = {
                'model': model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
    model_file = os.path.join(save_path,  f"{name}.pt")
    torch.save(state, model_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    return rt


if __name__ == '__main__':
    
    main()
    

