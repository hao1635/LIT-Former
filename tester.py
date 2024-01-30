import torch
from tqdm import tqdm
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from torchmetrics.functional import peak_signal_noise_ratio,structural_similarity_index_measure
import wandb
from util.util import get_logger,mkdirs,save_images,make_dir,crop_center,ssim_xy,compute_ssim,compute_rmse
import numpy as np
import pytorch_ssim




def mean_value(y_pred,y):
    y_pred=y_pred*3000-1000
    y_pred=torch.clip(y_pred,-160,240)
    y=y*3000-1000
    y=torch.clip(y,-160,240)
    return np.mean(abs(y-y_pred).detach().cpu().numpy())

def test(opt,model,loss_fn,testloader,device):
    import ipdb
    # ipdb.set_trace()
    if opt.mirror_padding is not None:
        opt.phase=opt.phase

    test_logger = get_logger('/mnt/data_jixie1/zhchen/MTDformer/test_results/'+opt.name+'.log')
    
    test_logger.info('start testing!')

    make_dir('/mnt/data_jixie1/zhchen/MTDformer/test_results/'+opt.name)
    make_dir('/mnt/data_jixie1/zhchen/MTDformer/test_results/'+opt.name+'/'+opt.phase+'-npy')

    Lambda=2

    up=torch.nn.Upsample(scale_factor=tuple([2.5,1,1]),mode='trilinear')

    ssim_loss = pytorch_ssim.SSIM3D(window_size = 11)
    #eval     
    test_running_psnr = []
    test_running_ssim3d=[]
    test_running_loss = 0   
    test_running_ssim2d=[]
    test_running_rmse=[]
    test_running_mean=[]

    model.eval()
    iters=0
    length=len(testloader.dataset) if opt.num_test==0 else opt.num_test

    with torch.no_grad():
        for x, y in tqdm(testloader):

            # if iters >= opt.num_test:  # only apply our model to opt.num_test images.
            #     break
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            b,c,d,h,w= y_pred.shape
            
            #test_loss = loss_fn(y_pred, y)
            test_loss = loss_fn(y_pred, y)+Lambda*(1-compute_ssim(y_pred, y))
            test_psnr=peak_signal_noise_ratio(y_pred,y)
            test_ssim3d=pytorch_ssim.ssim3D(y_pred,y)
            test_ssim2d=compute_ssim(y_pred, y)
            test_rmse=compute_rmse(y_pred, y)


            test_running_loss += test_loss.item()
            test_running_psnr.append(test_psnr.detach().cpu().numpy())
            test_running_ssim3d.append(test_ssim3d.detach().cpu().numpy())
            test_running_ssim2d.append(test_ssim2d.detach().cpu().numpy())
            test_running_rmse.append(test_rmse.detach().cpu().numpy())
            test_running_mean.append(mean_value(y_pred, y))
            
            if iters % 1== 0:  # save images to an HTML file
                test_logger.info('processing (%04d)-th image... ' % (iters))
                test_logger.info('(test_loss: %.6f, test_psnr: %.4f, test_ssim3d: %.4f,test_ssim2d: %.4f,test_rmse: %.6f) ' % (test_loss, test_psnr, test_ssim3d,test_ssim2d,test_rmse))
                #np.save('/mnt/data_jixie1/zhchen/MTDformer/test_results/'+opt.name+'/'+opt.phase+'-npy/y-pred-'+str('%02d' % iters),y_pred.cpu().detach().numpy())
                #save_images(torch.clip(y,0,1),root=save_images_root,phase='pred',index=iters,normalize=False)

            iters+=1

    average_test_loss = test_running_loss /length*opt.test_batch_size
    average_test_psnr= np.mean(test_running_psnr)
    average_test_ssim3d=np.mean(test_running_ssim3d)
    average_test_ssim2d=np.mean(test_running_ssim2d)
    average_test_rmse=np.mean(test_running_rmse)
    average_test_mean=np.mean(test_running_mean)
        
    test_message='(average_test_loss: %.6f, average_test_psnr: %.4f, average_test_ssim3d: %.4f,average_test_ssim2d:%.4f,average_test_rmse:%.6f,average_test_mean:%.4f) ' % (average_test_loss, average_test_psnr, average_test_ssim3d,average_test_ssim2d,average_test_rmse,average_test_mean)
    std_message='( std_test_psnr: %.4f, std_test_ssim3d: %.4f,std_test_ssim2d:%.4f,std_test_rmse:%.6f,std_test_mean:%.4f) ' % (np.std(test_running_psnr), np.std(test_running_ssim3d),np.std(test_running_ssim2d),np.std(test_running_rmse),np.std(test_running_mean))
    test_logger.info(test_message)
    test_logger.info(std_message)
    test_logger.info('finish!')




