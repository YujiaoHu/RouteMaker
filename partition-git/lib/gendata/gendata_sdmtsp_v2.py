import copy

import torch
from torch.utils.data import Dataset, DataLoader
import os


# from .ortools_generator import generate_ortools_data

class gendata_sdmtsp_ortools_for_test(Dataset):
    def __init__(self, cnum, anum, TL=2):
        super(gendata_sdmtsp_ortools_for_test, self).__init__()
        self.cnum = cnum
        self.anum = anum
        self.dataset = []
        self.datanum = 0
        objective = 'sdmtsp-cfd'
        datapath = os.path.join(os.getcwd(),
                                '../dataset/ortools/TL{}/{}/agent{}/city{}'
                                .format(TL, objective, self.anum, self.cnum))
        print(datapath)
        self.ortool_time = -1
        self.ave_tourlen = -1
        self.datapath = datapath
        self.load_data(datapath)
        self.datanum = len(self.dataset)
        assert self.datanum >= 0, "no ortools data for multi-depot mtsp"
        
    def __len__(self):
        self.datanum = len(self.dataset)
        return self.datanum

    def load_data(self, path):
        x = 0
        y = 0

        max_time = 0
        for i in range(201):
            filename = "sdMTSP-cfd_ortools_agent{}_city{}_num{}.pt" \
                .format(self.anum, self.cnum, i)
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath):
                data = torch.load(filepath)
                agent_coords = data['agent_coords']
                city_coords = data['city_coords']
                tourlen = data['tourlen']
                tusage = data['time']
                starts = data['start']
                ends = data['end']
                x += tusage
                self.dataset.append([agent_coords, city_coords, tourlen, starts, ends])
                y += torch.max(tourlen)

                if max_time < tusage:
                    max_time = tusage
            
        if len(self.dataset) > 0:
            self.ortool_time = x / len(self.dataset)
            self.ave_tourlen = y / len(self.dataset)
            print("\n ortools testing dataset for SDMTSP-CFD, anum = {}, cnum = {}".format(self.anum, self.cnum))
            print("\t\t average time usage in the dataset is {}".format(x / len(self.dataset)))
            print("\t\t average tour length is {}".format(y / len(self.dataset)))
            print("\t\t total data number is {}".format(len(self.dataset)))
            print("\n\t\t max_time = {}".format(max_time))
            print("****\n\n")

    def __getitem__(self, idx):
        const_L = 0.1
        agent_coords1 = self.dataset[idx][0]
        city_coords = self.dataset[idx][1]
        tourlen = self.dataset[idx][2]
        starts = self.dataset[idx][3]
        ends = self.dataset[idx][4]

        agent_coords2 = copy.deepcopy(agent_coords1)
        angle = 360 / self.anum * torch.pi / 180
        for a in range(self.anum):
            delta_x = torch.cos(torch.Tensor([angle * a])) * const_L
            delta_y = torch.sin(torch.Tensor([angle * a])) * const_L
            agent_coords2[a] = copy.deepcopy(agent_coords1[a] + torch.Tensor([delta_x, delta_y]))
        
        af = torch.cat([agent_coords2, agent_coords1], dim=1)
        cf = city_coords
        merge_coords = torch.cat([city_coords, agent_coords1, agent_coords1], dim=0)

        return af, cf, tourlen, merge_coords
 

class gendata_mdmtsp(Dataset): ## SDMTSP
    def __init__(self, problem='sdmtsp'):
        super(gendata_mdmtsp, self).__init__()
        assert problem == 'sdmtsp'
        self.problem = problem

    def __len__(self):
        return 1000

    def compute_dist(self, coordA, coordB):
        ca_square = torch.sum(coordA ** 2, dim=1, keepdim=True)
        cb_square = torch.sum(coordB ** 2, dim=1, keepdim=True)
        cross = -2 * torch.matmul(coordA, coordB.transpose(1, 0))
        dist = ca_square + cross + cb_square.transpose(1, 0)
        dist = torch.sqrt(dist)
        return dist

    def getitem(self, anum, cnum):
        # generate coordinates for agents' depots
        # depart from same depot, but ends with little bias
        
        const_L = 0.1
        base1 = torch.rand(1, 2)
        base2 = base1 * 0.7 + const_L + 0.05 
        depots_coord = base2.repeat(anum, 1)
        angle = 360 / anum * torch.pi / 180
        loc_coord = torch.zeros(anum, 2)
        for a in range(anum):
            delta_x = torch.cos(torch.Tensor([angle * a])) * const_L
            delta_y = torch.sin(torch.Tensor([angle * a])) * const_L
            loc_coord[a] = copy.deepcopy(base2 + torch.Tensor([delta_x, delta_y]))

        # generate coordinates for cities
        while True:
            city_mask = True
            city_coord = torch.rand(cnum, 2)
            dist1 = self.compute_dist(loc_coord, city_coord)
            dist2 = self.compute_dist(depots_coord, city_coord)
            dist3 = self.compute_dist(city_coord, city_coord)

            if (torch.sum(dist3 < 0.000001) > cnum or
                    torch.sum(dist1 < 0.000001) > 0 or
                    torch.sum(dist2 < 0.000001) > 0):
                print("torch.sum(dist3 < 0.000001) > cnum or torch.sum(dist1 < 0.000001) > 0 or "
                      "torch.sum(dist2 < 0.000001) > 0")
                continue
            else:
                break

        merge_coord = torch.cat([city_coord, depots_coord, depots_coord], dim=0)
        cf = city_coord
        af = torch.cat([loc_coord, depots_coord], dim=1)

        return merge_coord, cf, af


if __name__ == '__main__':
    gendata_mdmtsp(20, 4)
