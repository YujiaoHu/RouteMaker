import torch
from torch.utils.data import Dataset, DataLoader
import os


class Data_from_Real(Dataset):
    def __init__(self, scratch=True):
        super(Data_from_Real, self).__init__()
        self.dataset = []
        self.datanum = 0
        self.scratch = scratch

        # self.fnamelist = ['a280.tsp',  'bier127.tsp',
        #                   'd198.tsp', 'd493.tsp', 'fl417.tsp',
        #                   'ali535.tsp', 'att532.tsp', 'd1291.tsp', 'brd14051.tsp']

        self.fnamelist = ['ch130.tsp', 'ch150.tsp', 'eil101.tsp', 'gr202.tsp', 'kroA200.tsp',
                          'kroB150.tsp', 'rat195.tsp', 'rat575.tsp', 'rat783.tsp']

        for fname in self.fnamelist:
            datapath = os.path.join(os.getcwd(),
                                    '../dataset/real-world-TSPlib/{}'
                                    .format(fname))
            print(datapath)
            self.dataset.append(self.read_data(datapath))

        self.datanum = len(self.dataset)
        assert self.datanum >= 0, "no ortools data for multi-depot mtsp"

    def __len__(self):
        self.datanum = len(self.dataset)
        return self.datanum

    def read_data(self, datapath):
        coords = []
        with open(datapath) as file:
            fcontext = file.readlines()
            count_lines = len(fcontext)
            for c in range(count_lines - 1):
                temp = fcontext[c].split()
                if len(temp) == 3 and temp[1] != ':':
                    # print(fcontext[c].split())
                    order, x, y = fcontext[c].split()
                    coords.append(torch.tensor([float(x), float(y)]))
        coords = torch.stack(coords, dim=0)

        min_x = torch.min(coords[:, 0])
        min_y = torch.min(coords[:, 1])
        if min_x <= 0:
            coords[:, 0] = coords[:, 0] - min_x + 1
        if min_y <= 0:
            coords[:, 1] = coords[:, 1] - min_y + 1

        max_x = torch.max(coords[:, 0])
        max_y = torch.max(coords[:, 1])
        coords[:, 0] = coords[:, 0] / (max_x + 1)
        coords[:, 1] = coords[:, 1] / (max_y + 1)

        return coords

    def getitem(self, idx, anum):
        coords = self.dataset[idx]
        cnum = coords.size(0)
        assert cnum > anum

        if self.scratch is True:
            agent_coords = coords[:anum, :]
            city_coords = coords[anum:, :]
            af = torch.cat([agent_coords, agent_coords], dim=1)
            cf = city_coords
            merge_coords = torch.cat([city_coords, agent_coords, agent_coords], dim=0)
        else:
            loc_coords = coords[:anum, :]
            depot_coords = coords[cnum - anum:, :]
            city_coords = coords[anum: cnum - anum, :]
            af = torch.cat([loc_coords, depot_coords], dim=1)
            cf = city_coords
            merge_coords = torch.cat([city_coords, loc_coords, depot_coords], dim=0)

        # return af.unsqueeze(0), cf.unsqueeze(0), merge_coords.unsqueeze(0)
        return af, cf, merge_coords

    def compute_knn(self, cities):
        city_square = torch.sum(cities ** 2, dim=1, keepdim=True)
        city_square_tran = torch.transpose(city_square, 1, 0)
        cross = -2 * torch.matmul(cities, torch.transpose(cities, 1, 0))
        dist = city_square + city_square_tran + cross
        knn = torch.argsort(dist, dim=-1)
        knn = knn[:, :self.k]
        return knn
