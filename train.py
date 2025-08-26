import torch
from torch.autograd import Variable
import os
import numpy as np
from utils import OsJoin
import time
from utils import AverageMeter,calculate_accuracy

def train_epoch(epoch, fold_id, data_loader, model, criterion, optimizer,
                opt, epoch_logger, batch_logger, writer):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()
    accuracies = AverageMeter()

    writer = writer
    end_time  = time.time()
    labels_arr = torch.empty(4).cuda()
    pred_arr = torch.empty(4, 1).cuda()
    for i ,(inputs,labels) in enumerate(data_loader):
        torch.cuda.empty_cache()
        data_time.update(time.time()-end_time)
        labels = list(map(int, labels))
        inputs= (torch.unsqueeze(input,1) for input in inputs)
     #   inputs = inputs.type(torch.FloatTensor)
        inputs = (input.type(torch.FloatTensor) for input in inputs)
       # inputs = (input.type(torch.Float64) for input in inputs)
        if not opt.no_cuda:
            labels = torch.LongTensor(labels).cuda(non_blocking = True)
        inputs = (Variable(input) for input in inputs)
        #inputs = Variable(inputs)
        inputs = list(inputs)
        labels = Variable(labels)
        features_dict = ['ALFF','DFC', 'FA', 'FC']
        features_select = opt.features.split('_')
        indexs = (features_dict.index(feature) for feature in features_select)
        inputs_1 = (inputs[index] for index in indexs)
        #inputs_1=[inputs[0], inputs[2], inputs[3]]
        inputs = [list(inputs_1), labels]
        loss, outputs = model(inputs)
        acc = calculate_accuracy(outputs, labels)

        losses.update(loss.data,inputs[0][0].size(0))
        accuracies.update(acc, inputs[0][0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time()-end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i+1,
            'iter': (epoch-1)*len(data_loader)+(i-1),
            'loss': round(losses.val.item(), 4),
            'acc': round(accuracies.val.item(), 4),
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch, i + 1, len(data_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, acc=accuracies))
        _, pred = outputs.topk(k=1, dim=1, largest=True)
        pred_arr = torch.cat([pred_arr, pred], dim=0)
        labels_arr = torch.cat([labels_arr, labels], dim=0)
    print('prediction :', end=' ')
    for i in range(4, len(pred_arr)):
        print('%d\t'%(pred_arr[i]), end='')
    print('\nlabel    :', end=' ')
    for i in range(4, len(labels_arr)):
        print('%d\t'%(labels_arr[i]), end='')
    print('\n')
    labels_arr = torch.empty(4).cuda()
    pred_arr = torch.empty(4, 1).cuda()
    epoch_logger.log({
        'epoch': epoch,
        'loss': round(losses.avg.item(), 4),
        'acc': round(accuracies.avg.item(), 4),
        'lr': optimizer.param_groups[0]['lr']
    })

    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/accuracy', accuracies.avg, epoch)

    if opt.save_weight:
        if epoch % opt.checkpoint == 0:
            save_dir =OsJoin(opt.result_path, opt.data_type, opt.model_name + str(opt.model_depth),
                             'weights_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = OsJoin(save_dir,
                        '{}{}_weights_fold{}_epoch{}.pth'.format(opt.model_name, opt.model_depth, fold_id, epoch))
            states = {
                'fold': fold_id,
                'epoch': epoch,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_path)



