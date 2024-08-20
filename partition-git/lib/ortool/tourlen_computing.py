import torch
import multiprocessing
from .ortool_entrance import entrance as ortools
import gc

try:
    multiprocessing.set_start_method('spawn', force=True)
    # print(multiprocessing.get_start_method())
except RuntimeError:
    pass


# multiprocessing.set_start_method('spawn')

def classify_tourform(inputs):
    anum = len(inputs)

    tourlen = torch.zeros(anum)
    tourset = []
    for a in range(anum):
        atour = inputs[a][0]
        xcoord = inputs[a][1]
        # print(atour, xcoord)
        # use ortool compute, return tour length
        # print("starting planning, ", xcoord.size()[0])
        tourlen[a], singleTour = ortools(xcoord)
        # print('ending planning')
        tourset.append(atour[singleTour])
    return [tourlen, tourset]


def classify_tourform_wout_tours(coordsets):
    anum = len(coordsets)
    tourlen = torch.zeros(anum)
    for a in range(anum):
        xcoord = coordsets[a]
        tourlen[a], singleTour = ortools(xcoord)
    return tourlen


def get_tour_len_wout_tours(tour, coords, anum, parallel=True):
    device = tour.device
    batch_size, instance_num, cnum = tour.size()
    reshape_tours = tour.contiguous().view(batch_size * instance_num, cnum)
    reshape_coords = (coords.unsqueeze(1).repeat(1, instance_num, 1, 1)
                      .view(batch_size * instance_num, cnum + 2 * anum, 2))

    multi_idxs = []
    for b in range(batch_size * instance_num):
        agent_comp_set = []
        for a in range(anum):
            x = torch.tensor([cnum + a, cnum + anum + a]).to(device)
            aidxs = reshape_tours[b].eq(a)
            atour = torch.nonzero(aidxs).view(-1)
            x = torch.cat([x, atour], dim=0)
            gather_coords = torch.gather(reshape_coords[b], 0, x.unsqueeze(1).repeat(1, 2))
            agent_comp_set.append(gather_coords.cpu())
        multi_idxs.append(agent_comp_set)

    if parallel is True:
        pool = multiprocessing.Pool(processes=10)
        result = pool.map(classify_tourform_wout_tours, multi_idxs)
        pool.close()
        pool.join()
    else:
        result = []
        for idxs in multi_idxs:
            result.append(classify_tourform_wout_tours(idxs))

    tourlen = torch.stack(result, dim=0).view(batch_size, instance_num, anum)
    return tourlen.clone().to(device)


def get_tour_len(tour, coords, anum, parallel=True):
    device = tour.device
    batch_size, cnum, dims = coords.size()
    cnum = cnum - 2 * anum
    number_samples = tour.size(1)

    multi_idxs = []
    for b in range(batch_size):
        for n in range(number_samples):
            agent_comp_set = []
            for a in range(anum):
                x = torch.tensor([cnum + a, cnum + anum + a]).to(device)
                aidxs = tour[b, n].eq(a)
                atour = torch.nonzero(aidxs).view(-1)
                x = torch.cat([x, atour], dim=0)
                gather_coords = torch.gather(coords[b], 0, x.unsqueeze(1).repeat(1, 2))
                agent_comp_set.append([x.cpu(), gather_coords.cpu()])
            multi_idxs.append(agent_comp_set)

    if parallel is True:
        pool = multiprocessing.Pool(processes=10)
        result = pool.map(classify_tourform, multi_idxs)
        pool.close()
        pool.join()
    else:
        result = []
        for idxs in multi_idxs:
            result.append(classify_tourform(idxs))

    tourlen = []
    tourset = []
    for b in range(batch_size):
        nsampleTour = []
        for n in range(number_samples):
            tourlen.append((result[b * number_samples + n][0]).to(device))
            nsampleTour.append(result[b * number_samples + n][1])
        tourset.append(nsampleTour)
    tourlen = torch.stack(tourlen, dim=0).view(batch_size, number_samples, anum)

    # # single process
    # test_result = []
    # for b in range(batch_size):
    #     inputs = [index[b], coords[b], anum]
    #     test_result.append(classify_tourform(inputs))
    # test_result = torch.stack(test_result, dim=0)
    # print(" test result:", test_result-result)
    # print("result:", result[0])
    return tourlen.clone().to(device), tourset


def get_tour_len_singleTrack(tour, coords, anum):
    device = tour.device
    batch_size, cnum, dims = coords.size()
    number_samples = tour.size(1)

    # samples_tours = [[[] for n in range(number_samples)] for b in range(batch_size)]
    tourlen = []
    tourset = []
    for b in range(batch_size):
        nsampleTour = []
        for n in range(number_samples):
            agent_tour = []
            # print("partition:", tour[b,n])
            for a in range(anum):
                aidxs = tour[b, n].eq(a)
                atour = torch.nonzero(aidxs).view(-1).cpu() + 1
                agent_tour.append(atour)
                # print("b = {}, n = {}, a = {}, atour = {}".format(b, n, a, atour))
                # samples_tours[b][n].append(torch.tensor(atour).to(device))
            # print("agent tour", agent_tour)
            result = classify_tourform([agent_tour, coords[b].cpu(), anum])
            tourlen.append(result[0])
            nsampleTour.append(result[1])
        tourset.append(nsampleTour)
    tourlen = torch.stack(tourlen, dim=0).view(batch_size, number_samples, anum)
    return tourlen.to(device), tourset
