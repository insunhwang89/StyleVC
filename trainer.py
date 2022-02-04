
import warnings
import os, sys
warnings.filterwarnings("ignore")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # 상위 폴더내 파일 참조
print("gpu id ----------------> ", os.environ["CUDA_VISIBLE_DEVICES"])

# pytorch
import torch
import torch.nn as nn

# model
import model.hparams as hp
from model.model import Model
from model.discriminator import Discriminator

from model.loss import LossFunction, LSGANLoss
from model.train import train
from model.validation import validation
from online_inference import run_online_inference

# utils
from utils.utils import save_checkpoint, load_checkpoint, copy_code
from utils.writer import get_writer
from utils.data_utils import prepare_dataloaders
from utils.scheduler import ScheduledOptimizer

# multi-processing
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

# temp
# from inference import run_inference
def main(rank, num_gpus, batch_size): 

    if num_gpus > 1:
        init_process_group(
            backend=hp.dist_backend, init_method=hp.dist_url, world_size=hp.world_size * num_gpus, rank=rank)
    device = torch.device('cuda:{:d}'.format(rank))
    torch.cuda.set_device(rank)

    # Load model
    model_g = Model(hp).cuda(rank)    
    model_d = Discriminator(hp).cuda(rank)
    if num_gpus > 1:
        model_g = DistributedDataParallel(model_g, device_ids=[rank])
        model_d = DistributedDataParallel(model_d, device_ids=[rank])

    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=hp.lr, betas=[0.8, 0.99], eps=1e-9)
    model_g, optimizer_g, iteration = load_checkpoint(hp, model_g, optimizer_g, device, option="G")
    scheduler_g = ScheduledOptimizer(optimizer_g, hp.lr, 4000, iteration)
    
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=2e-4, betas=[0.9, 0.98], eps=1e-9)
    model_d, optimizer_d, _ = load_checkpoint(hp, model_d, optimizer_d, device, option="D")

    scaler = torch.cuda.amp.GradScaler()     
    
    # loss function
    criterion = LossFunction(hp)
    adversarial_loss = LSGANLoss()

    # loader
    train_loader, val_loader, train_sampler = prepare_dataloaders(hp, rank, num_gpus, hp.num_workers, batch_size)    
    epoch_offset = max(0, int(iteration / len(train_loader)))

    writer = None
    if rank == 0:
        writer = get_writer(hp.output_directory, f'{hp.log_directory}')
        copy_code(hp.output_directory, f'{hp.log_directory}')

    print(f'Model training start!!! {hp.log_directory}')
    for epoch in range(epoch_offset, 10000):

        if rank == 0:
            print(f'Epoch: {epoch + 1}, lr: {hp.lr}') 

        if num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, (batch1, batch2) in enumerate(train_loader): # X_(A,i), X_(A,k)

            model_g.train()
            model_d.train()
            iteration += 1

            train(
                hp, model_g, optimizer_g,  
                model_d, optimizer_d, adversarial_loss,
                criterion, writer, iteration, batch1, batch2, device, rank, scaler)   


            if rank == 0:
                if iteration % hp.iters_per_checkpoint == 0:    
                    save_checkpoint(
                        model_g, optimizer_g, iteration, num_gpus,
                        filepath=f'{hp.output_directory}/{hp.log_directory}', option="G")
                    save_checkpoint(
                        model_d, optimizer_d, iteration, num_gpus,
                        filepath=f'{hp.output_directory}/{hp.log_directory}', option="D")

                if iteration % hp.iters_per_validation == 0:
                    validation(hp, model_g, criterion, model_d, adversarial_loss, 
                        val_loader, iteration, device, writer)  

                if iteration % hp.iters_per_online_inference == 0:
                    run_online_inference(hp, model_g, writer, iteration)
            
            if iteration == hp.stop_iteration:
                break

        if iteration == hp.stop_iteration:
            break

    print(f'Training finish!!!')
    
if __name__ == '__main__':

    torch.manual_seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        num_gpus = torch.cuda.device_count()
        batch_size = int(hp.train_batch_size / num_gpus)
        print('Num  GPU :', num_gpus)
        print('Batch size per GPU :', batch_size)

    if num_gpus > 1:
        mp.spawn(main, nprocs=num_gpus, args=(num_gpus, batch_size,))
    else:
        main(0, num_gpus, batch_size)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='VCTK') 
    # parser.add_argument('--dataset_name', default='VCTK_16K')
    parser.add_argument('--log_dir', default='StyleVC_VCTK_test01')
    a = parser.parse_args()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='NIKL') 
    # parser.add_argument('--dataset_name', default='NIKL_22K')
    # parser.add_argument('--log_dir', default='RCVC_NIKL_test01')
    # a = parser.parse_args()

    # need more improvement code
    # update hparams ######################################################################################
    # hp.dataset = a.dataset
    # hp.dataset_path = a.dataset_name
    hp.log_directory = a.log_dir

    # # dataset
    # if a.dataset == "VCTK":
    #     hp.male = ['p226', 'p227', 'p232', 'p237', 'p241', 'p243', 'p245', 'p246', 'p247', 'p251',
    #             'p252', 'p254', 'p255', 'p256', 'p258', 'p259', 'p260', 'p263', 'p270', 'p271', 
    #             'p272', 'p273', 'p274', 'p275', 'p278', 'p279', 'p281', 'p284', 'p285', 'p286',
    #             'p287', 'p292', 'p298', 'p302', 'p304', 'p311', 'p316', 'p326', 'p334', 'p345', 
    #             'p347', 'p360', 'p363', 'p364', 'p374', 'p376']  
    #     hp.female = ['p225', 'p228', 'p229', 'p230', 'p231', 'p233', 'p234', 'p236', 'p238', 'p239',
    #             'p240', 'p244', 'p248', 'p249', 'p250', 'p253', 'p257', 'p261', 'p262', 'p264',
    #             'p265', 'p266', 'p267', 'p268', 'p269', 'p276', 'p277', 'p280', 'p282', 'p283', 
    #             'p288', 'p293', 'p294', 'p295', 'p297', 'p299', 'p300', 'p301', 'p303', 'p305', 
    #             'p306', 'p307', 'p308', 'p310', 'p312', 'p313', 'p314', 'p317', 'p318', 'p323', 
    #             'p329', 'p330', 'p333', 'p335', 'p336', 'p339', 'p340', 'p341', 'p343', 'p351', 
    #             'p361', 'p362']
    # elif a.dataset == "NIKL":
    #     hp.male = ['mv01', 'mv02', 'mv03', 'mv04', 'mv05', 'mv06', 'mv07', 'mv08', 'mv09', 'mv10', 
    #             'mv11', 'mv12', 'mv13', 'mv15', 'mv16', 'mv17', 'mv19', 'mv20', 'mw01', 'mw02',
    #             'mw03', 'mw04', 'mw05', 'mw06', 'mw07'] 
    #     hp.female = ['fv01', 'fv02', 'fv03', 'fv04', 'fv05', 'fv06', 'fv07', 'fv08', 'fv09', 'fv10', 
    #             'fv11', 'fv12', 'fv13', 'fv14', 'fv15', 'fv16', 'fv17', 'fv18', 'fv19', 'fv20',
    #             'fx01', 'fx02', 'fx03', 'fx04', 'fx05']

    # hp.speakers = hp.male + hp.female
    # hp.n_speakers = len(hp.speakers)
    # hp.speaker_ids = dict(zip(hp.speakers, range(len(hp.speakers))))

    #################################################################################################

    torch.manual_seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    
    main()