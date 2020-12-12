from collections import defaultdict

import torch


def parse_data(data):
    pids = data[1]
    params_3d = data[4]

    return pids, params_3d


def smpl_statistics(dataloader):
    params_dict = defaultdict(list)

    for data in dataloader:
        pids, params_3d = parse_data(data)

        for idx in range(pids.size(0)):
            params_dict[idx].append(params_3d[idx].unsqueeze(0))

    print('Start collecting statistics of smpl params...')
    params_tensor_dict = dict() # concate params for per person identity
    params_tensor_overall = None    # concate params of the whole data loader
    for pid in params_dict:
        param_tentor = torch.cat(params_dict[pid])
        params_tensor_dict[pid] = param_tentor

        if params_tensor_overall is None:
            params_tensor_overall = param_tentor
        else:
            params_tensor_overall = torch.cat([params_tensor_overall, param_tentor])

    print('Start writing the results...')
    output_path = '/home/jiaxing/mpips-smplify_public_v2/smplify_public/results/prcc/smplify_params_statistics.txt'
    with open(output_path, 'w') as fp:
        overall_mean = torch.mean(params_tensor_overall, dim=0)
        overall_var = torch.var(params_tensor_overall, dim=0)

        fp.write('{:^12}:{}, {:^12}:{}'.format('Over Mean', str(overall_mean.numpy()),
                                               'Over Var', str(overall_var.numpy())))

        for pid in params_dict:
            mean = torch.mean(params_tensor_dict[pid], dim=0)
            var = torch.var(params_tensor_dict[pid], dim=0)
            fp.write('{:^9}{:^3}:{}, {:^9}{:^3}:{}'.format('Pid Mean ', str(pid), str(mean.numpy()),
                                               'Pid Var ', str(pid), str(var.numpy())))
