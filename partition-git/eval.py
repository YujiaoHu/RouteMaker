import os
import torch
from options import get_options
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from lib.transformerlayers.dystructure import DyTransformerMSTP as mdmtsp
from lib.gendata.gendata_sdmtsp_v2 import gendata_mdmtsp as gendata_sdmtsp
from lib.gendata.gendata_mix import gendata_mdmtsp as gendata_cfa_cfd
# from lib.ortool.tourlen_computing import get_tour_len_singleTrack as orhelp
from lib.ortool.tourlen_computing import get_tour_len as orhelp
from lib.ortool.tourlen_computing import get_tour_len_wout_tours as orhelp_wout_tours
import time
from torch.distributions import Categorical
from lib.gendata.read_real_data import Data_from_Real
from lib.decode.beamsearch import beam_search_decoding
from lib.decode.move_from_longest import move_from_longest_decoding
import multiprocessing
from tqdm import tqdm

try:
    multiprocessing.set_start_method('spawn', force=True)
    # print(multiprocessing.get_start_method())
except RuntimeError:
    pass


def tourlen_computing(inputs, tour, anum, notours=True):
    # inputs: coordinates [batch. cnum, 2]
    # tour: [batch, number_samples, cnum_1]
    with torch.no_grad():
        if notours is True:
            tourlen = orhelp_wout_tours(tour, inputs, anum)
            samples_tours = None
        else:
            tourlen, samples_tours = orhelp(tour, inputs, anum)
    return tourlen, samples_tours


class TrainModleDyMTSP(nn.Module):
    def __init__(self, load=True,
                 _modelpath=os.path.join(os.getcwd(), "../savemodel-correct"),
                 anum=2,
                 cnum=20,
                 _device=torch.device('cuda:0'),
                 lr=1e-5,
                 train_instance=10,
                 entropy_coeff=0.1,
                 obj='minmax',
                 problem='sdmtsp'):
        super(TrainModleDyMTSP, self).__init__()
        
        self.load = load
        self.device = _device
        self.modelpath = _modelpath
        self.anum = anum
        self.cnum = cnum
        self.lr = lr
        self.clip_argv = 3
        self.train_instance = train_instance
        self.entropy_coeff = entropy_coeff

        self.obj = obj
        self.problem = problem
        
        if self.problem in ['cfa', 'cfd']:
            self.gendata_mdmtsp = gendata_cfa_cfd
        else:
            self.gendata_mdmtsp = gendata_sdmtsp

        self.icnt = 0
        self.epoch = 0

        self.model_name = ("git_mixed_sdmtsp_obj={}_problem={}_lr={}_entropy={}_trainIns={}"
                           .format(self.obj, self.problem, self.lr,
                                   self.entropy_coeff, self.train_instance))

        self.model = mdmtsp(2, 4, 1024, 5)
        self.model.to(self.device)

        self.writer = SummaryWriter('../runs_obj={}_problem={}/{}'
                                    .format(self.obj, self.problem, self.model_name))
        self.modelfile = os.path.join(self.modelpath, '{}.pt'.format(self.model_name))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if load:
            print("loading model:{}".format(self.modelfile))
            if os.path.exists(self.modelfile):
                checkpoint = torch.load(self.modelfile, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Model loaded")
            else:
                print("No Model loaded")

    def rl_loss_computing(self, logits, tourlen, partition):
        maxtourlen = torch.max(tourlen, dim=2)[0]
        baselineTourlen = torch.mean(maxtourlen, dim=1, keepdim=True)
        advantage = maxtourlen - baselineTourlen  # advantage:[number_samples]
        # print("advantage ", advantage)
        temp_partition = partition.permute(0, 2, 1)
        probsloss = torch.gather(logits, 2, temp_partition)  # [cnum, number_samples]
        probsloss = torch.sum(torch.log(probsloss), dim=1)

        entropy = torch.mean(Categorical(logits).entropy())  # Tensor:[batch, req_num]

        loss = torch.mean(probsloss * advantage) - self.entropy_coeff * entropy

        return loss, entropy.item()


    def mdmtsp_eval_with_imp_decoding(self, cnum, anum, moveTL, batch_size=10, valnum=200):
        self.model.eval()
        
        valset = self.gendata_mdmtsp(self.problem)
        
        print("testing on dataset with anum = {},  cnum = {}, problem = {} with datanum = {} using improvement decoding"
              "...".format(anum, cnum, self.problem, valnum))
        
        assert valnum % batch_size == 0
        maxiter = valnum // batch_size

        batch_opt_len = []
        batch_tusage = []
        with torch.no_grad():
            for it in tqdm(range(maxiter)):
                batch_merge_coord = []
                batch_cf = []
                batch_af = []
                for b in range(batch_size):
                    merge_coord, cf, af = valset.getitem(anum, cnum)
                    batch_merge_coord.append(merge_coord)
                    batch_af.append(af)
                    batch_cf.append(cf)
                cf = torch.stack(batch_cf, dim=0)
                af = torch.stack(batch_af, dim=0)
                merge_coord = torch.stack(batch_merge_coord, dim=0)
                
                af, cf, merge_coord = af.to(self.device), cf.to(self.device), merge_coord.to(self.device)

                start_time = time.time()
                probs, partition = self.model(af, cf, maxsample=True, instance_num=1)
                end_time = time.time()
                moveTL = moveTL - (end_time - start_time) / batch_size
                opt_len, tusage, opt_partition = move_from_longest_decoding(
                    probs, merge_coord, 0.7, moveTL,
                    on_policy=True,
                    auto_pthrd=True, parallel=True)
                batch_tusage.append(tusage + (end_time - start_time) / batch_size)
                batch_opt_len.append(opt_len)
                print("  {}/{} finished with mfl decoding, opt_len = {}, tusage = {}"
                      .format(it, maxiter, torch.mean(opt_len), torch.mean(tusage)))
        batch_opt_len = torch.cat(batch_opt_len, dim=0)
        batch_tusage = torch.cat(batch_tusage, dim=0)
        
        print("anum = {}, cnum = {}, valnum = {}, decode moveTL = {}".format(anum, cnum, valnum, moveTL))
        print('Average time usage : {}'.format(torch.mean(batch_tusage)))
        print("Average tour length: ", torch.mean(batch_opt_len))
            
            
    def mdmtsp_eval_with_sampling_decoding(self, cnum, anum, decode_sample=1, batch_size=20, valnum=200):
        self.model.eval()

        valset = self.gendata_mdmtsp(self.problem)

        print("testing on dataset with anum = {},  cnum = {}, problem = {} with datanum = {} using sampling decoding"
              "...".format(anum, cnum, self.problem, valnum))
        
        assert valnum % batch_size == 0
        maxiter = valnum // batch_size

        net_len = []
        net_time = []
        with torch.no_grad():
            for it in tqdm(range(maxiter)):
                batch_merge_coord = []
                batch_cf = []
                batch_af = []
                for b in range(batch_size):
                    merge_coord, cf, af = valset.getitem(anum, cnum)
                    batch_merge_coord.append(merge_coord)
                    batch_af.append(af)
                    batch_cf.append(cf)
                cf = torch.stack(batch_cf, dim=0)
                af = torch.stack(batch_af, dim=0)
                merge_coord = torch.stack(batch_merge_coord, dim=0)
                
                af, cf, merge_coord = af.to(self.device), cf.to(self.device), merge_coord.to(self.device)

                start_time = time.time()
                if decode_sample == 1:
                    probs, partition = self.model(af, cf, maxsample=True, instance_num=1)
                else:
                    probs, partition = self.model(af, cf, maxsample=False, instance_num=decode_sample)
                partition = partition.permute(0, 2, 1)
                # tours:[batch, self.instance_num, self.anum]
                tourlen, tours = tourlen_computing(merge_coord, partition, anum)
                end_time = time.time()
                maxlen_overins = torch.max(tourlen, dim=-1)[0]
                net_len.append(torch.min(maxlen_overins, dim=-1)[0])
                net_time.append(torch.Tensor([end_time - start_time]) / batch_size)

            net_len = torch.cat(net_len, dim=0)
            net_time = torch.cat(net_time, dim=0)

            print("anum = {}, cnum = {}, valnum = {}, decode sampling number = {}".format(anum, cnum, valnum, decode_sample))
            print('Average time usage : {}'.format(torch.mean(net_time)))
            print("Average tour length: ", torch.mean(net_len))
        print("test ends")


def main():
    opts = get_options()
    anum = opts.anum
    cnum = opts.cnum
    batch_size = opts.batch_size
    device = torch.device(opts.cuda)
    # device = torch.device('cpu')
    lr = opts.lr
    trainIns = opts.trainIns
    modelpath = opts.modelpath
    entropy_coeff = opts.entropy_coeff
    objective = opts.obj
    problem = opts.problem

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    tsp = TrainModleDyMTSP(_modelpath=modelpath,
                           anum=anum,
                           cnum=cnum,
                           _device=device,
                           lr=lr,
                           train_instance=trainIns,
                           entropy_coeff=entropy_coeff,
                           obj=objective,
                           problem=problem
                           )
    
    decode_strategy = opts.decode_strategy
    if decode_strategy == 'greedy':
        sampling_num = 1
        tsp.mdmtsp_eval_with_sampling_decoding(
            cnum, anum, decode_sample=1, batch_size=batch_size, valnum=5*batch_size)
    if decode_strategy == 'sampling':
        sampling_num = opts.sampling_number
        tsp.mdmtsp_eval_with_sampling_decoding(
            cnum, anum, decode_sample=sampling_num, batch_size=batch_size, valnum=5*batch_size)
    if decode_strategy == 'improvement':
        moveTL = opts.moveTL
        tsp.mdmtsp_eval_with_imp_decoding(cnum, anum, moveTL, batch_size, valnum=5*batch_size)


if __name__ == '__main__':
    print(os.getcwd())
    main()
