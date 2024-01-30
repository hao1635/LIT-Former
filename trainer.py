import torch
from tqdm import tqdm
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from torchmetrics.functional import peak_signal_noise_ratio,structural_similarity_index_measure
import wandb
from util.util import get_logger,mkdirs,save_images,ssim_xy,compute_ssim,compute_rmse,compute_psnr2D
import os
import math
import pytorch_ssim
import ipdb




def train(opt,model,optimizer,lr_scheduler,loss_fn,trainloader,testloader,device):
    
    if opt.use_wandb:
        wandb.init(project='litformer_review',name=opt.name)
        wandb.watch(model)
    train_logger = get_logger(opt.checkpoints_dir+'/'+opt.name+'/train.log')
    save_images_root='./results/'+opt.name
    mkdirs(save_images_root)  
    train_logger.info(model)
    train_logger.info('start training!')

    train_total_iters = 0
    val_total_iters = 0
    Lambda=2
    for epoch in tqdm(range(opt.epochs)):
        train_length=len(trainloader.dataset)
        val_length=len(testloader.dataset)
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        running_loss = 0
        running_psnr3d=0
        running_pnsr2d=0
        running_ssim3d=0
        running_ssim2d=0
        

        model.train()

        if epoch>1:
            for x, y in tqdm(trainloader):

                x, y = x.to(device), y.to(device)

                y_pred = model(x)
                
                train_total_iters += 1
                epoch_iter += 1
                

                train_loss = loss_fn(y_pred, y)+Lambda*(1-compute_ssim(y_pred, y))
                train_psnr3d=peak_signal_noise_ratio(y_pred,y)
                # train_psnr2d=compute_psnr2D(y_pred, y)
                train_ssim3d=pytorch_ssim.ssim3D(y_pred,y)
                train_ssim2d=compute_ssim(y_pred, y)
                

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                #lr_scheduler.step()
                with torch.no_grad():
                    running_loss += train_loss.item()
                    running_psnr3d += train_psnr3d
                    running_ssim2d+=train_ssim2d
                    running_ssim3d+=train_ssim3d
                    

                if train_total_iters % opt.print_freq == 0:
                    message='(epoch: %d, iters: %d,epoch_loss: %.4f, train_psnr3d: %.4f,train_ssim3d: %.4f,train_ssim2d:.%.4f) ' % (epoch, epoch_iter,train_loss, train_psnr3d,train_ssim3d,train_ssim2d)
                    print(message)
                    if opt.use_wandb:
                        wandb.log({"train_loss": train_loss,
                                    'train_psnr':train_psnr3d,
                                    'train_ssim':train_ssim3d} )
                    
            epoch_loss = running_loss/train_length*opt.train_batch_size
            epoch_psnr3d= running_psnr3d/train_length*opt.train_batch_size
            epoch_ssim3d=running_ssim3d/train_length*opt.train_batch_size
            epoch_ssim2d=running_ssim2d/train_length*opt.train_batch_size
            

            #train_message='(epoch: %d, iters: %d,epoch_loss: %.4f, train_psnr: %.4f, train_ssim: %.4f) ' % (epoch, epoch_iter,epoch_loss, epoch_psnr, epoch_ssim)
            train_logger.info('Epoch: [{}/{}],epoch_loss: {:.6f}, train_psnr3d: {:.4f},train_ssim3d: {:.4f},train_ssim2d:{:.4f}'.format(epoch ,opt.epochs, epoch_loss, epoch_psnr3d, epoch_ssim3d, epoch_ssim2d))

        #eval  
        # if epoch>30:
        print('validation:')   
        test_running_psnr3d = 0
        test_running_ssim3d=0
        test_running_loss = 0 
        test_running_ssim2d=0
        test_running_rmse=0

        #ipdb.set_trace()
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(testloader):
                val_total_iters+=1
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                #test_loss=loss_fn(y_pred, y)
                test_loss=loss_fn(y_pred, y)+Lambda*(1-compute_ssim(y_pred, y))
                test_psnr3d=peak_signal_noise_ratio(y_pred,y)
                test_ssim3d=pytorch_ssim.ssim3D(y_pred,y)
                test_ssim2d=compute_ssim(y_pred, y)
                test_rmse=compute_rmse(y_pred, y)
                


                test_running_loss += test_loss.item()
                test_running_psnr3d+=test_psnr3d
                test_running_ssim3d+=test_ssim3d
                test_running_ssim2d+=test_ssim2d
                test_running_rmse+=test_rmse
                    

                if val_total_iters % (opt.print_freq/2) == 0:
                    #print('(test_loss: %.6f, test_psnr: %.3f, test_ssim: %.5f) ' % (test_loss, test_psnr, test_ssim))
                    if opt.use_wandb:
                        wandb.log({"test_loss": test_loss,
                                    'test_psnr':test_psnr3d,
                                    'test_ssim':test_ssim3d} ) 
        epoch_test_loss = test_running_loss /val_length
        epoch_test_psnr3d= test_running_psnr3d/val_length
        epoch_test_ssim3d=test_running_ssim3d/val_length
        epoch_test_ssim2d=test_running_ssim2d/val_length 
        epoch_test_rmse=test_running_rmse/val_length      

        train_logger.info('val:Epoch: [{}/{}],epoch_loss: {:.6f}, val_psnr3d: {:.4f}, val_ssim3d: {:.4f},val_ssim2d: {:.4f},test_rmse: {:.6f}'.format(epoch , opt.epochs, epoch_test_loss, epoch_test_psnr3d,epoch_test_ssim3d,epoch_test_ssim2d,epoch_test_rmse))

        
        lr_scheduler.step()
        if opt.use_wandb:
            wandb.log({"epoch_train_loss": epoch_loss,
                        'epoch_train_psnr':epoch_psnr,
                        'epoch_train_ssim':epoch_ssim,
                        "epoch_test_loss":epoch_test_loss,
                        'epoch_test_psnr':epoch_test_psnr3d,
                        'epoch_test_ssim':epoch_test_ssim3d,
                        'epoch':epoch
                        } )
#save model
        if epoch%5==0:
            if len(opt.gpu_ids)>1:
                static_dict=model.module.state_dict()
            else:
                static_dict=model.state_dict()
            torch.save(static_dict,opt.checkpoints_dir+'/'+opt.name+'/{}_trainloss_{:.6f}_train_psnr{:.3f}_train_ssim.pth'.format(epoch,epoch_loss,epoch_psnr,epoch_ssim))
    train_logger.info('finish training!')




