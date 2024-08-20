import copy

import torch
from torch.utils.data import Dataset, DataLoader
import os


# from .ortools_generator import generate_ortools_data


class gendata_mdmtsp(Dataset):
    def __init__(self, problem='cfa'):
        super(gendata_mdmtsp, self).__init__()
        assert problem in ['cfd', 'cfa']
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
        while True:
            depots_coord = torch.rand(anum, 2)
            dist = self.compute_dist(depots_coord, depots_coord)
            if torch.sum(dist < 0.001) > anum:
                print("torch.sum(dist < 0.001) > anum")
                continue
            else:
                break

        if self.problem == 'cfd':
            loc_coord = copy.deepcopy(depots_coord)
        else:            
            while True:
                loc_coord = torch.rand(anum, 2)
                dist1 = self.compute_dist(loc_coord, depots_coord)
                dist2 = self.compute_dist(loc_coord, loc_coord)
                if torch.sum(dist1 < 0.001) > 0 or torch.sum(dist2 < 0.001) > anum:
                    print("torch.sum(dist1 < 0.001) > 0 or torch.sum(dist2 < 0.001) > 0")
                    continue
                else:
                    break

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

        merge_coord = torch.cat([city_coord, loc_coord, depots_coord], dim=0)
        cf = city_coord
        af = torch.cat([loc_coord, depots_coord], dim=1)

        return merge_coord, cf, af


if __name__ == '__main__':
    gendata_mdmtsp(20, 4)
