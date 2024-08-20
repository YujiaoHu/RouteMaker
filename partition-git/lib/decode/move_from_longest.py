import copy
import multiprocessing
import time

import torch
from lib.ortool.ortool_entrance import entrance as ortools
from lib.ortool.tourlen_computing import get_tour_len as orhelp
from lib.ortool.tourlen_computing import get_tour_len_wout_tours as orhelp_wout_tours

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
            tourlen = orhelp_wout_tours(tour, inputs, anum, parallel=False)
            samples_tours = None
        else:
            tourlen, samples_tours = orhelp(tour, inputs, anum, parallel=False)
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


def move_from_longest_for_each_batch(inputs):
    probs, merge_coords, TL, pthrd, notour, on_policy, auto_pthrd = inputs

    def has_been_explored(object, objlist):
        stack_objlist = torch.stack(objlist, dim=0)
        x = torch.sum(torch.abs(stack_objlist - object.unsqueeze(0)), dim=1)
        if 0 in x:
            return True
        else:
            return False

    cnum, anum = probs.size()
    start_time = time.time()
    max_value, max_partition = torch.max(probs, dim=1)
    tourlen, _ = tourlen_computing(merge_coords.unsqueeze(0),
                                   max_partition.unsqueeze(0).unsqueeze(0),
                                   anum, notours=notour)
    tourlen = tourlen.view(-1)
    partition = max_partition.view(-1)
    greedy_len = torch.max(tourlen)

    opt_partition = partition
    opt_len = greedy_len

    stack_partition = [partition]
    stack_tourlen = [tourlen]

    explored_partition = []
    stack_explore_num = 0
    while len(stack_partition) > 0:
        if time.time() - start_time >= TL:
            # print("     time limitation with TL = {}, opt_len = {}".format(TL, opt_len))
            break
        # if stack_explore_num > 30:
        #     # print(" has explored 30 partitions, but no improvements, opt_len = {}".format(opt_len))
        #     break

        stack_explore_num = stack_explore_num + 1
        h = torch.stack(stack_tourlen, dim=0)
        cnt = torch.argmin(torch.max(h, dim=1)[0])
        const_partition = stack_partition.pop(cnt)
        tourlen = stack_tourlen.pop(cnt)

        explored_partition.append(copy.deepcopy(const_partition))

        longest_agent = torch.argmax(tourlen)
        longest_len = torch.max(tourlen)

        aidxs = const_partition.eq(longest_agent)
        longest_tour = torch.nonzero(aidxs).view(-1)

        partition_prob = torch.gather(probs, 1, const_partition.unsqueeze(1)).view(-1)
        longest_probs = torch.gather(partition_prob, 0, longest_tour)

        if on_policy is True:
            values, indices = torch.sort(longest_probs)
            if auto_pthrd is True:
                pthrd = max(values.median(), 0.7) + 0.05
                # if pthrd < 0.75:
                #     indices = torch.randperm(longest_tour.size(0))
                #     pthrd = 1.1
                # x = torch.sum(torch.less(longest_probs, pthrd))
                # print(" the pthrd of agent {} is {}, exchange number is {}/{}, len = {}"
                #       .format(longest_agent, pthrd, x.item(), longest_tour.size(0), longest_len))
        else:
            indices = torch.randperm(longest_tour.size(0))
            pthrd = 1.1
        for idx in indices:
            partition = copy.deepcopy(const_partition)
            if longest_probs[idx] > pthrd:
                # print("    longest_probs[idx] = {} > {}".format(longest_probs[idx], pthrd))
                break
            if time.time() - start_time >= TL:
                # print("     time limitation with TL = {}".format(TL))
                break
            corder = longest_tour[idx]
            partition[corder] = -1

            spa_len = get_tour_len_wout_tours_for_specific_agent(
                partition, merge_coords, cnum, anum, longest_agent)
            mid_len = copy.deepcopy(tourlen)
            mid_len[longest_agent] = spa_len
            for a in range(anum):
                if time.time() - start_time >= TL:
                    # print("     time limitation with TL = {}".format(TL))
                    break
                if a != longest_agent:
                    tlen = copy.deepcopy(mid_len)
                    partition[corder] = a
                    if not has_been_explored(partition, explored_partition):
                        spa_len = get_tour_len_wout_tours_for_specific_agent(
                            partition, merge_coords, cnum, anum, a)
                        tlen[a] = spa_len
                        x = torch.max(tlen.view(-1))
                        # tourlen1, _ = tourlen_computing(merge_coords.unsqueeze(0),
                        #                                 partition.unsqueeze(0).unsqueeze(0),
                        #                                 anum, notours=True)
                        # assert torch.sum(torch.abs(tourlen1.view(-1) - tlen)) == 0, 'error 142'
                        if x < longest_len:
                            stack_partition.append(copy.deepcopy(partition))
                            stack_tourlen.append(tlen.view(-1))

                            # print("stak partition increase: ", len(stack_partition))
                            if x < opt_len:
                                # print("\n ***** opt_len: from {} to {} \n".format(
                                #     opt_len.item(), x.item()))
                                opt_partition = copy.deepcopy(partition)
                                opt_len = x
                                stack_explore_num = 0
                        else:
                            explored_partition.append(copy.deepcopy(partition))
    return [greedy_len, opt_len, time.time() - start_time, opt_partition]


def move_from_longest_decoding(batch_probs, batch_merge_coords, pthrd, TL, on_policy, auto_pthrd, parallel=True):
    device = batch_probs.device

    batch_size, cnum, anum = batch_probs.size()

    batch_greedy_len = torch.zeros(batch_size)
    batch_opt_len = torch.zeros(batch_size)
    batch_tusage = torch.zeros(batch_size)
    batch_opt_partition = []

    if parallel is False:
        for b in range(batch_size):
            greedy_len, opt_len, tusage, opt_partition = move_from_longest_for_each_batch(
                [batch_probs[b], batch_merge_coords[b], TL, pthrd, True, on_policy, auto_pthrd])
            # print("\n\n")
            # print("greedy_len, opt_len, tusage", greedy_len, opt_len, tusage)
            batch_greedy_len[b] = greedy_len
            batch_opt_len[b] = opt_len
            batch_tusage[b] = tusage
            batch_opt_partition.append(opt_partition)
    else:
        multi_idxs = []
        for b in range(batch_size):
            multi_idxs.append([batch_probs[b].cpu(), batch_merge_coords[b].cpu(),
                               TL, pthrd, True, on_policy, auto_pthrd])

        pool = multiprocessing.Pool(processes=10)
        result = pool.map(move_from_longest_for_each_batch, multi_idxs)
        pool.close()
        pool.join()

        for b in range(batch_size):
            batch_greedy_len[b] = result[b][0]
            batch_opt_len[b] = result[b][1]
            batch_tusage[b] = result[b][2]
            batch_opt_partition.append(result[b][3])

    return batch_opt_len.to(device), batch_tusage.to(device), torch.stack(batch_opt_partition, dim=0)
