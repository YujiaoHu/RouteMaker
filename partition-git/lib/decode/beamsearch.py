import copy
import time

import torch
from lib.ortool.ortool_entrance import entrance as ortools
from lib.ortool.tourlen_computing import get_tour_len as orhelp
from lib.ortool.tourlen_computing import get_tour_len_wout_tours as orhelp_wout_tours


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


def get_tour_len_wout_tours_for_specific_agent(tour, coords, cnum, anum, spa):
    '''
        tour: [cnum]
        coords: [cnum + 2 * anum, 2]
        anum: int
        spa: int
    '''
    device = coords.device
    x = torch.tensor([cnum + spa, cnum + anum + spa]).to(device)
    aidxs = tour.eq(spa)
    atour = torch.nonzero(aidxs).view(-1)
    x = torch.cat([x, atour], dim=0)
    gather_coords = torch.gather(coords, 0, x.unsqueeze(1).repeat(1, 2))
    spa_tlen, traj = ortools(gather_coords.cpu())
    return spa_tlen


def beam_search_decodeing_for_each_batch(probs, merge_coords, beam, TL, pthrod=0.8, notour=True):
    cnum, anum = probs.size()
    start_time = time.time()
    max_value, max_partition = torch.max(probs, dim=1)
    tourlen, _ = tourlen_computing(merge_coords.unsqueeze(0),
                                   max_partition.unsqueeze(0).unsqueeze(0),
                                   anum, notours=notour)
    indices = torch.sort(max_value.view(-1))[1]
    beam_tours = max_partition.unsqueeze(0)
    beam_tourlen = tourlen.view(1, -1)
    greedy_len = torch.max(tourlen.view(-1))

    cnt = 0
    min_max_len = greedy_len
    for idx in indices:
        if max_value[idx] > pthrod:
            break
        if time.time() - start_time >= TL:
            # print("     time limitation with TL = {}".format(TL))
            break
        temp_tours = []
        temp_tourlen = []
        pnum = beam_tours.size(0)
        for p in range(pnum):
            if time.time() - start_time >= TL:
                # print("     time limitation with TL = {}".format(TL))
                break
            temp_tours.append(copy.deepcopy(beam_tours[p]))
            temp_tourlen.append(copy.deepcopy(beam_tourlen[p]))

            current_partition = beam_tours[p]
            belong_agent = current_partition[idx]

            current_partition[idx] = -1
            spa_len = get_tour_len_wout_tours_for_specific_agent(
                current_partition, merge_coords, cnum, anum, belong_agent)

            mid_len = copy.deepcopy(beam_tourlen[p])
            mid_len[belong_agent] = spa_len
            for a in range(anum):
                if time.time() - start_time >= TL:
                    # print("     time limitation with TL = {}".format(TL))
                    break
                if a != belong_agent:
                    tlen = copy.deepcopy(mid_len)
                    current_partition[idx] = a
                    spa_len = get_tour_len_wout_tours_for_specific_agent(
                        current_partition, merge_coords, cnum, anum, a)
                    tlen[a] = spa_len
                    x = torch.max(tlen.view(-1))
                    if x < 1.2 * min_max_len:
                        temp_tours.append(copy.deepcopy(h))
                        temp_tourlen.append(tlen.view(-1))
                    if min_max_len < x:
                        min_max_len = x

        temp_tours = torch.stack(temp_tours, dim=0)
        temp_tourlen = torch.stack(temp_tourlen, dim=0)
        if len(temp_tourlen) <= beam:
            beam_tours = temp_tours
            beam_tourlen = temp_tourlen
        else:
            max_temp_tourlen = torch.max(temp_tourlen, dim=1)[0]
            values, index = torch.sort(torch.Tensor(max_temp_tourlen))
            beam_tourlen = temp_tourlen[list(index[:beam].cpu().numpy())]
            beam_tours = temp_tours[list(index[:beam].cpu().numpy())]
    return greedy_len, torch.min(torch.max(beam_tourlen, dim=1)[0]), time.time() - start_time


def beam_search_decoding(batch_probs, batch_merge_coords, beam, pthrod, TL=3600):
    batch_size, cnum, anum = batch_probs.size()
    batch_beam_len = torch.zeros(batch_size)
    batch_tusage = torch.zeros(batch_size)

    for b in range(batch_size):
        greedy_len, beam_len, tusage = (
            beam_search_decodeing_for_each_batch(batch_probs[b],
                                                 batch_merge_coords[b],
                                                 beam, TL, pthrod))
        # print("greedy_len, beam_len, tusage", greedy_len, beam_len, tusage)
        # print("\n\n")
        batch_beam_len[b] = beam_len
        batch_tusage[b] = tusage
    return batch_beam_len, batch_tusage
