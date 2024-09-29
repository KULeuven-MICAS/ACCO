#%%
import pickle, os, time
import itertools
from classes.stages.DepthFirstStage import DFMultiOutputVilation
import numpy as np

from classes.stages.MainInputParserStages import parse_workload_from_path
from classes.workload.layer_node import LayerNode, InputLayerNode
from classes.depthfirst.data_copy_layer import DataCopyLayer

#%%
# ['Meta_prototype', 'Edge_TPU_like', 'Tesla_NPU_like']
hw_wl_name = 'Meta_prototype'

root_dir = './result_pickle_files_misc_camready'
root_dir_cache = './result_pickle_files_misc_camready_cache'
home_dir = f'{root_dir}/test_{hw_wl_name}/test_mp_energy'
home_dir_old_hw = f'{root_dir}/test_{hw_wl_name}/test_mp_latency'
log_dir = f'{root_dir}_log/test_{hw_wl_name}/test_mp_energy'
log_dir_old_hw = f'{root_dir}_log/test_{hw_wl_name}/test_mp_latency'
cache_dir = f'{root_dir_cache}/test_{hw_wl_name}/test_mp_energy'
cache_dir_old_hw = f'{root_dir_cache}/test_{hw_wl_name}/test_mp_latency'

total_layers_dict = {
    'seldnet': 8,
    'stftCNN': 12,
    'mobilenetv1': 27,
    'resnet18': 26
}

multi_output_layer_idx_list_dict = {
    'stftCNN': [],
    'mobilenetv1': [],
    'resnet18': [1, 4, 7, 10, 13, 16, 19, 22]
}

df_tcn_substack_cut_max_dict = {
    'seldnet': 2,
    'stftCNN': 3,
    'mobilenetv1': 5,
    'resnet18': 5
}

workload = 'workload_stftCNN_fixed_unrolling_oxoy_025_16x'
workload_folder = 'inputs'

# df_tcn_global_initial_dilation_list = [1, 2, 4, 8]
df_tcn_global_initial_dilation_list = [-1]
df_tcn_frame_amount = 1

# df_caching_list = [[True, True], [True, False], [False, True], [False, False]]
df_caching_list = [[True, True]]

var_names = locals()

#%%
# Fetch all the original data
df_tcn_global_initial_dilation_list = [-1]
df_y_tilesizes = [1, 2, 4, 8, 16, 32]
df_x_tilesizes = [1, 2, 4, 8, 16, 32, 64, 128]

df_y_tilesizes = [1, 2, 4, 8, 16, 32, 64]
df_x_tilesizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
df_caching_list = [[True, True]]
df_tcn_substack_cut_max = 2

result_ensemble_dict_layer_summary = {}
result_ensemble_dict_frame_total = {}
result_ensemble_dict_cache_total = {}
result_ensemble_dict_cache_stable = {}

result_ensemble_dict_layer_summary_old_hw = {}
result_ensemble_dict_frame_total_old_hw = {}
result_ensemble_dict_cache_total_old_hw = {}
result_ensemble_dict_cache_stable_old_hw = {}

for df_tcn_global_initial_dilation in df_tcn_global_initial_dilation_list:
    for df_tilesize_x in df_x_tilesizes:
        for df_tilesize_y in df_y_tilesizes:
            for [df_horizontal_caching, df_vertical_caching] in df_caching_list:
                path_format = f'{workload}_dil-{df_tcn_global_initial_dilation}_fr-{df_tcn_frame_amount}'
                file_format = f'{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}'
                key_format = f'{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}_dil-{df_tcn_global_initial_dilation}_fr-{df_tcn_frame_amount}'

                result_ensemble_dict_layer_summary[key_format] = {}
                with open(f'{log_dir}/{path_format}/{file_format}_layer_summary.log', 'r') as fp:
                    for line in fp.readlines():
                        _line_list = line.replace('\n', '').split(',')
                        result_ensemble_dict_layer_summary[key_format][f'{_line_list[0]},{_line_list[1]}'] = _line_list[2:]

                result_ensemble_dict_frame_total[key_format] = {}
                with open(f'{log_dir}/{path_format}/{file_format}_frame_total.log', 'r') as fp:
                    for line in fp.readlines():
                        _line_list = line.replace('\n', '').split(',')
                        result_ensemble_dict_frame_total[key_format][f'{_line_list[0]},{_line_list[1]}'] = _line_list[2:]

                result_ensemble_dict_cache_total[key_format] = {}
                with open(f'{cache_dir}/{path_format}/{file_format}_total_cache.log', 'r') as fp:
                    for line in fp.readlines():
                        _line_list = line.replace('\n', '').split(',')
                        result_ensemble_dict_cache_total[key_format][f'{_line_list[0]},{_line_list[1]}'] = _line_list[4:]

                result_ensemble_dict_cache_stable[key_format] = {}
                with open(f'{cache_dir}/{path_format}/{file_format}_stable_cache.log', 'r') as fp:
                    for line in fp.readlines():
                        _line_list = line.replace('\n', '').split(',')
                        result_ensemble_dict_cache_stable[key_format][f'{_line_list[0]},{_line_list[1]}'] = _line_list[4:]

                # another hw (actually another opt_strategy results)
                result_ensemble_dict_layer_summary_old_hw[key_format] = {}
                with open(f'{log_dir_old_hw}/{path_format}/{file_format}_layer_summary.log', 'r') as fp:
                    for line in fp.readlines():
                        _line_list = line.replace('\n', '').split(',')
                        result_ensemble_dict_layer_summary_old_hw[key_format][f'{_line_list[0]},{_line_list[1]}'] = _line_list[2:]

                result_ensemble_dict_frame_total_old_hw[key_format] = {}
                with open(f'{log_dir_old_hw}/{path_format}/{file_format}_frame_total.log', 'r') as fp:
                    for line in fp.readlines():
                        _line_list = line.replace('\n', '').split(',')
                        result_ensemble_dict_frame_total_old_hw[key_format][f'{_line_list[0]},{_line_list[1]}'] = _line_list[2:]

                result_ensemble_dict_cache_total_old_hw[key_format] = {}
                with open(f'{cache_dir_old_hw}/{path_format}/{file_format}_total_cache.log', 'r') as fp:
                    for line in fp.readlines():
                        _line_list = line.replace('\n', '').split(',')
                        result_ensemble_dict_cache_total_old_hw[key_format][f'{_line_list[0]},{_line_list[1]}'] = _line_list[4:]

                result_ensemble_dict_cache_stable_old_hw[key_format] = {}
                with open(f'{cache_dir_old_hw}/{path_format}/{file_format}_stable_cache.log', 'r') as fp:
                    for line in fp.readlines():
                        _line_list = line.replace('\n', '').split(',')
                        result_ensemble_dict_cache_stable_old_hw[key_format][f'{_line_list[0]},{_line_list[1]}'] = _line_list[4:]

                # different opt_strategy should share the same cache size in each scenario
                assert (result_ensemble_dict_cache_total_old_hw[key_format] == result_ensemble_dict_cache_total[key_format])
                assert (result_ensemble_dict_cache_stable_old_hw[key_format] == result_ensemble_dict_cache_stable[key_format])

                # if _probing_id < _probing_fin and [df_horizontal_caching, df_vertical_caching] == [True, True]:
                #     with open(f'{home_dir}/{path_format}/{file_format}_{str(df_stack_cuts)}.pkl', 'rb') as fp:
                #         _probing_cmes.append(pickle.load(fp))
                #     _probing_id += 1

#%%
# freedom dive for cut-tuple candidates
df_tcn_global_initial_dilation_list = [-1]

df_y_tilesizes = [1, 2, 4, 8, 16, 32, 64]
df_x_tilesizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
df_caching_list = [[True, True]]

df_tcn_substack_cut_max = df_tcn_substack_cut_max_dict[workload.split('_')[1]]
total_layers = total_layers_dict[workload.split('_')[1]]
last_layer_idx = total_layers - 1
multi_output_layer_idx_list = multi_output_layer_idx_list_dict[workload.split('_')[1]]

df_stack_cuts = []

# get the weight cut index
df_tcn_global_initial_dilation = -1
df_tcn_frame_amount = 1
df_tilesize_x = 4
df_tilesize_y = 4
df_horizontal_caching = True
df_vertical_caching = True

cme_probe = []

path_format = f'{workload}_dil-{df_tcn_global_initial_dilation}_fr-{df_tcn_frame_amount}'
file_format = f'{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}'
var_names[f'{workload}_weight_cut_cases'] = []
with open(f'{home_dir}/{path_format}/{file_format}_[].pkl', 'rb') as fp:
    cme_ensemble = pickle.load(fp)
    for cme in cme_ensemble:
        if cme[1] not in var_names[f'{workload}_weight_cut_cases']:
            var_names[f'{workload}_weight_cut_cases'].append(cme[1])

        # just probe on one cme to debug
        if cme[2][0]['cut_before'] == 5 and cme[2][0]['cut_after'] == 8:
            cme_probe.append(cme)

# generate the full cut id candidates
cut_id_candidate_list_new_hw = []
cut_id_candidate_list_old_hw = []

layer_idx_list = list(range(last_layer_idx+1)) if last_layer_idx != total_layers - 1 else list(range(total_layers))
if -1 in layer_idx_list:
    layer_idx_list.remove(-1)
_MAX = last_layer_idx if last_layer_idx != total_layers - 1 else None
if _MAX == None:
    layer_idx_list.remove(last_layer_idx)
for idx in df_stack_cuts:
    layer_idx_list.remove(idx)

for weight_cut in var_names[f'{workload}_weight_cut_cases']:
    cut_id_extra = list(i for i in weight_cut if i < last_layer_idx)
    _layer_idx_list = []
    for idx in layer_idx_list:
        if idx not in cut_id_extra:
            _layer_idx_list.append(idx)

    cut_id_candidate_list = []
    for i in range(df_tcn_substack_cut_max + 1):
        cut_id_candidate_list += (list(itertools.combinations(_layer_idx_list, i)))

    for item in set(tuple(sorted(set(i_tuple + tuple(df_stack_cuts) + tuple(cut_id_extra)))) for i_tuple in cut_id_candidate_list):
        try:
            for cut_before, cut_after in zip([None] + list(item), list(item) + [_MAX]):
                _cut_before = cut_before if cut_before is not None else -1
                _cut_after = cut_after if cut_after is not None else last_layer_idx
                cut_idx_list = list(range(_cut_before + 1, _cut_after + 1))
                for multi_output_node_idx in multi_output_layer_idx_list:
                    if multi_output_node_idx in cut_idx_list and _cut_after not in multi_output_layer_idx_list:
                        print(f'Found invalid cut range [{cut_before},{cut_after}] as it hits {multi_output_node_idx} and [{cut_after}] not in {multi_output_layer_idx_list}')
                        raise DFMultiOutputVilation
        except:
            continue
        else:
            cut_id_candidate_list_new_hw.append(list(item))
            cut_id_candidate_list_old_hw.append(list(item))

#%%
# the substack-wise latency/energy ranking for different test on total & real-time frames

# there is a sweet spot as
# 1) the offload/reload for certain layer&mode is almost constant
# 2) the benefits from the parallelism saturate at high tiling sizes
# 3) the cache size saturate at higher tiling size && earlier at deep sub-stacks

df_tcn_global_initial_dilation_list = [-1]
df_y_tilesizes = [1, 2, 4, 8, 16, 32]
df_x_tilesizes = [1, 2, 4, 8, 16, 32, 64, 128]
df_y_tilesizes = [1, 2, 4, 8, 16, 32, 64]
df_x_tilesizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

df_caching_list = [[True, True]]

assert (cut_id_candidate_list_new_hw == cut_id_candidate_list_old_hw)
cut_id_candidate_list = cut_id_candidate_list_new_hw

# sub-stack-wise latency/energy ranking, together with the offloading/reloading overhead
case_study_substack_latency_dict_total = {}
case_study_substack_energy_dict_total = {}
case_study_substack_latency_dict_realtime = {}
case_study_substack_energy_dict_realtime = {}
case_study_substack_cache_dict_total = {}
case_study_substack_cache_dict_realtime = {}

case_study_substack_data_offload_dict_total = {}
case_study_substack_data_reload_dict_total = {}
case_study_substack_data_offload_dict_realtime = {}
case_study_substack_data_reload_dict_realtime = {}

for df_cut in cut_id_candidate_list_new_hw:
    for cut_before, cut_after in zip([None] + df_cut, df_cut + [_MAX]):
        _dict_latency = {}
        _dict_energy = {}
        _dict_data_offload = {}
        _dict_data_reload = {}

        for (key, value) in result_ensemble_dict_frame_total.items():
            key = key.replace('_dil--1_fr-1', '')
            _dict_latency[key+'_energy'] = [int(i) for i in value[f'{cut_before},{cut_after}'][0:6:2]] # full, offload, reload
            _dict_energy[key+'_energy'] = [int(i) for i in value[f'{cut_before},{cut_after}'][1:6:2]]
            _dict_data_offload[key + '_energy'] = int(value[f'{cut_before},{cut_after}'][6])
            _dict_data_reload[key + '_energy'] = int(value[f'{cut_before},{cut_after}'][7])

        case_study_substack_latency_dict_total[f'{cut_before},{cut_after}'] = dict(sorted(_dict_latency.items(), key=lambda x: x[1][0]))
        case_study_substack_energy_dict_total[f'{cut_before},{cut_after}'] = dict(sorted(_dict_energy.items(), key=lambda x: x[1][0]))
        case_study_substack_data_offload_dict_total[f'{cut_before},{cut_after}'] = _dict_data_offload
        case_study_substack_data_reload_dict_total[f'{cut_before},{cut_after}'] = _dict_data_reload

        _dict_latency = {}
        _dict_energy = {}
        _dict_data_offload = {}
        _dict_data_reload = {}

        for (key, value) in result_ensemble_dict_layer_summary.items():
            if key.split('_')[1] == '1': # tile-y == 1
                key = key.replace('_dil--1_fr-1', '')
                _dict_latency[key + '_energy'] = [int(i) for i in value[f'{cut_before},{cut_after}'][1:6:2]]
                _dict_energy[key + '_energy'] = [int(i) for i in value[f'{cut_before},{cut_after}'][0:6:2]]
                _dict_data_offload[key + '_energy'] = int(value[f'{cut_before},{cut_after}'][12])
                _dict_data_reload[key + '_energy'] = int(value[f'{cut_before},{cut_after}'][14])

        case_study_substack_latency_dict_realtime[f'{cut_before},{cut_after}'] = dict(sorted(_dict_latency.items(), key=lambda x: x[1][0]))
        case_study_substack_energy_dict_realtime[f'{cut_before},{cut_after}'] = dict(sorted(_dict_energy.items(), key=lambda x: x[1][0]))
        case_study_substack_data_offload_dict_realtime[f'{cut_before},{cut_after}'] = _dict_data_offload
        case_study_substack_data_reload_dict_realtime[f'{cut_before},{cut_after}'] = _dict_data_reload

        # Sub-stack cache size info: passed assertion that the cache info is shared by old_hw
        _dict_cache = {}
        for (key, value) in result_ensemble_dict_cache_total.items():
            key = key.replace('_dil--1_fr-1', '')
            _dict_cache[key] = [float(i) for i in value[f'{cut_before},{cut_after}'][2:]] # peak_f, weights
        case_study_substack_cache_dict_total[f'{cut_before},{cut_after}'] = dict(sorted(_dict_cache.items(), key=lambda x: x[1][0], reverse=True))

        _dict_cache = {}
        for (key, value) in result_ensemble_dict_cache_stable.items():
            if key.split('_')[1] == '1':
                key = key.replace('_dil--1_fr-1', '')
                _dict_cache[key] = [float(i) for i in value[f'{cut_before},{cut_after}'][2:]] # peak_f, weights
        case_study_substack_cache_dict_realtime[f'{cut_before},{cut_after}'] = dict(sorted(_dict_cache.items(), key=lambda x: x[1][0], reverse=True))

#%%
# explore the end-to-end
_case_study_substack_dict_latency = case_study_substack_latency_dict_total
_case_study_substack_dict_energy = case_study_substack_energy_dict_total
_case_study_substack_dict_cache = case_study_substack_cache_dict_total


case_study_combination_latency = {}
case_study_combination_energy = {}
case_study_combination_EDP = {}

_latency_0 = []
_latency_1 = []
_latency_2 = []
_energy_0 = []
_energy_1 = []
_energy_2 = []

for df_cut in cut_id_candidate_list_new_hw:
    _substack_ranges = []
    _substack_keys = []
    _substacks = {}
    _tmp_case_study_combination_latency = {
        'latency': float('inf'),
        'energy': -1,
        'peak_f' : -1,
        'strategy' : None
    }
    _tmp_case_study_combination_energy = {
        'latency': -1,
        'energy': float('inf'),
        'peak_f': -1,
        'strategy': None
    }

    _tmp_case_study_combination_EDP = {
        'EDP': float('inf'),
        'latency': -1,
        'energy': -1,
        'peak_f': -1,
        'strategy': None
    }

    for cut_before, cut_after in zip([None] + df_cut, df_cut + [_MAX]):
        _substack_ranges.append(f'{cut_before},{cut_after}')
        _substack_keys.append(_case_study_substack_dict_latency[f'{cut_before},{cut_after}'].keys())
    _substacks_combinations = itertools.product(*_substack_keys)

    # _len = 0
    for keys in _substacks_combinations:
        _strategy = {}
        _latency = 0
        _latency_delta = 0
        _energy = 0
        _energy_delta = 0
        _cache = 0.0

        for i, key in enumerate(keys):
            _strategy[_substack_ranges[i]] = key

            _data = _case_study_substack_dict_latency[_substack_ranges[i]][key]
            _latency += _data[0]
            _latency_delta += _data[1] + _data[2]

            _data = _case_study_substack_dict_energy[_substack_ranges[i]][key]
            _energy += _data[0]
            _energy_delta += _data[1] + _data[2]

            _cache_key = '_'.join(list(key.split('_'))[0:4])
            _cache += _case_study_substack_dict_cache[_substack_ranges[i]][_cache_key][0]

        if _cache < 1.0:
            _latency = _latency - _latency_delta
            _energy = _energy - _energy_delta

        _EDP = _latency * _energy

        if _latency < _tmp_case_study_combination_latency['latency']:
            _tmp_case_study_combination_latency['latency'] = _latency
            _tmp_case_study_combination_latency['energy'] = _energy
            _tmp_case_study_combination_latency['peak_f'] = _cache
            _tmp_case_study_combination_latency['strategy'] = _strategy

        if _energy < _tmp_case_study_combination_energy['energy']:
            _tmp_case_study_combination_energy['latency'] = _latency
            _tmp_case_study_combination_energy['energy'] = _energy
            _tmp_case_study_combination_energy['peak_f'] = _cache
            _tmp_case_study_combination_energy['strategy'] = _strategy

        if _EDP < _tmp_case_study_combination_EDP['EDP']:
            _tmp_case_study_combination_EDP['EDP'] = _EDP
            _tmp_case_study_combination_EDP['latency'] = _latency
            _tmp_case_study_combination_EDP['energy'] = _energy
            _tmp_case_study_combination_EDP['peak_f'] = _cache
            _tmp_case_study_combination_EDP['strategy'] = _strategy

        # dirty statistics for stftCNN-16x design space
        if len(df_cut) == 0 and _latency < 5.2 * 1e6 and _energy < 0.8 * 1e9 and len(_latency_0) < 1008:
            _latency_0.append(_latency)
            _energy_0.append(_energy)
        elif len(df_cut) == 1 and _latency < 5.2 * 1e6 and _energy < 0.8 * 1e9 and len(_latency_1) < 2200:
            _latency_1.append(_latency)
            _energy_1.append(_energy)
        elif len(df_cut) == 2 and _latency < 5.2 * 1e6 and _energy < 0.8 * 1e9 and len(_latency_2) < 2200:
            _latency_2.append(_latency)
            _energy_2.append(_energy)

    case_study_combination_latency[_tmp_case_study_combination_latency['latency']] = {
        'df_cut': df_cut,
        'energy': _tmp_case_study_combination_latency['energy'],
        'strategy': _tmp_case_study_combination_latency['strategy'],
        'peak_f': _tmp_case_study_combination_latency['peak_f'],
    }

    case_study_combination_energy[_tmp_case_study_combination_energy['energy']] = {
        'df_cut': df_cut,
        'latency': _tmp_case_study_combination_energy['latency'],
        'strategy': _tmp_case_study_combination_energy['strategy'],
        'peak_f': _tmp_case_study_combination_energy['peak_f'],
    }

    case_study_combination_EDP[_tmp_case_study_combination_EDP['EDP']] = {
        'df_cut': df_cut,
        'latency': _tmp_case_study_combination_EDP['latency'],
        'energy': _tmp_case_study_combination_EDP['energy'],
        'strategy': _tmp_case_study_combination_EDP['strategy'],
        'peak_f': _tmp_case_study_combination_EDP['peak_f'],
    }

case_study_combination_latency = dict(sorted(case_study_combination_latency.items(), key=lambda x: x[0]))
case_study_combination_energy = dict(sorted(case_study_combination_energy.items(), key=lambda x: x[0]))
case_study_combination_EDP = dict(sorted(case_study_combination_EDP.items(), key=lambda x: x[0]))

#%%
# case study: show design space
substack_length = 2
_latency = []
_energy = []

for (key, value) in case_study_combination_latency.items():
    if len(value['df_cut']) == substack_length:
        _latency.append(key)
        _energy.append(value['energy'])

for (key, value) in case_study_combination_energy.items():
    if len(value['df_cut']) == substack_length:
        _energy.append(key)
        _latency.append(value['latency'])

for (key, value) in case_study_combination_EDP.items():
    if len(value['df_cut']) == substack_length:
        _latency.append(value['latency'])
        _energy.append(value['energy'])
#%%
# case study: explicit substack cut + fixed tile cases + same tiling for all substacks
# explore the end-to-end
_case_study_substack_dict_latency = case_study_substack_latency_dict_total
_case_study_substack_dict_energy = case_study_substack_energy_dict_total
_case_study_substack_dict_cache = case_study_substack_cache_dict_total
_case_study_substack_dict_data_offload = case_study_substack_data_offload_dict_total
_case_study_substack_dict_data_reload = case_study_substack_data_reload_dict_total

# df_cut = []
df_cut = [5, 9]

_tile_scenario = []
_peak_f = []
_data_extra = []
_latency_real = []
_latency_total = []
_energy_real = []
_energy_total = []


_substack_ranges = []
_substack_keys = []
_substacks = {}
_tmp_case_study_combination_latency = {
    'latency': float('inf'),
    'energy': -1,
    'peak_f' : -1,
    'strategy' : None
}
_tmp_case_study_combination_energy = {
    'latency': -1,
    'energy': float('inf'),
    'peak_f': -1,
    'strategy': None
}

for cut_before, cut_after in zip([None] + df_cut, df_cut + [_MAX]):
    _substack_ranges.append(f'{cut_before},{cut_after}')

_substacks_combinations = [[f'{i}_{i}_True_True_energy'] * (len(df_cut) + 1) for i in [1, 2, 4, 8, 16, 32, 64]]
# _len = 0
for keys in _substacks_combinations:
    _strategy = {}
    _latency = 0
    _latency_delta = 0
    _energy = 0
    _energy_delta = 0
    _cache = 0.0
    _data_offload_reload = 0

    _tile_scenario.append(keys[0])

    for i, key in enumerate(keys):
        _strategy[_substack_ranges[i]] = key

        _data = _case_study_substack_dict_latency[_substack_ranges[i]][key]
        _latency += _data[0]
        _latency_delta += _data[1] + _data[2]

        _data = _case_study_substack_dict_energy[_substack_ranges[i]][key]
        _energy += _data[0]
        _energy_delta += _data[1] + _data[2]

        _data_offload_reload += _case_study_substack_dict_data_offload[_substack_ranges[i]][key] + _case_study_substack_dict_data_reload[_substack_ranges[i]][key]

        _cache_key = '_'.join(list(key.split('_'))[0:4])
        _cache += _case_study_substack_dict_cache[_substack_ranges[i]][_cache_key][0]

    _peak_f.append(_cache)
    _data_extra.append(_data_offload_reload / 8 / 1024 ** 2)
    _latency_total.append(_latency)
    _energy_total.append(_energy)

    if _cache < 1.0:
        _latency = _latency - _latency_delta
        _energy = _energy - _energy_delta

    _latency_real.append(_latency)
    _energy_real.append(_energy)

#%%
# the best combination for one cut_id_list across (MemAllocation + TileSize): key = score
# TODO: cross-compare the combination of opt_strategy and the sorted metric here; Is the minLatency's energy always larger than the minEnergy's;
case_study_best_combination_latency_total : dict = {}
for df_cut in cut_id_candidate_list_new_hw:
    _latency = 0
    _energy = 0
    _substacks = {}
    for cut_before, cut_after in zip([None] + df_cut, df_cut + [_MAX]):
        _substack_key = list(case_study_substack_latency_dict_total[f'{cut_before},{cut_after}'].keys())[0]
        _latency += case_study_substack_latency_dict_total[f'{cut_before},{cut_after}'][_substack_key]
        _energy += case_study_substack_energy_dict_total[f'{cut_before},{cut_after}'][_substack_key]
        _substacks[f'{cut_before},{cut_after}'] = _substack_key
    case_study_best_combination_latency_total[','.join(str(i) for i in df_cut)] = {
        'latency': _latency,
        'energy' : _energy,
        'strategy' : _substacks
    }

case_study_best_combination_latency_total = dict(sorted(case_study_best_combination_latency_total.items(), key=lambda x: x[1]['latency']))

case_study_best_combination_energy_total: dict = {}
for df_cut in cut_id_candidate_list_new_hw:
    _latency = 0
    _energy = 0
    _substacks = {}
    for cut_before, cut_after in zip([None] + df_cut, df_cut + [_MAX]):
        _substack_key = list(case_study_substack_energy_dict_total[f'{cut_before},{cut_after}'].keys())[0]
        _latency += case_study_substack_latency_dict_total[f'{cut_before},{cut_after}'][_substack_key]
        _energy += case_study_substack_energy_dict_total[f'{cut_before},{cut_after}'][_substack_key]
        _substacks[f'{cut_before},{cut_after}'] = _substack_key
    case_study_best_combination_energy_total[','.join(str(i) for i in df_cut)] = {
        'latency': _latency,
        'energy' : _energy,
        'strategy' : _substacks
    }

case_study_best_combination_energy_total = dict(sorted(case_study_best_combination_energy_total.items(), key=lambda x: x[1]['energy']))

case_study_best_combination_latency_realtime : dict = {}
for df_cut in cut_id_candidate_list_new_hw:
    _latency = 0
    _energy = 0
    _substacks = {}
    for cut_before, cut_after in zip([None] + df_cut, df_cut + [_MAX]):
        _substack_key = list(case_study_substack_latency_dict_realtime[f'{cut_before},{cut_after}'].keys())[0]
        _latency += case_study_substack_latency_dict_realtime[f'{cut_before},{cut_after}'][_substack_key]
        _energy += case_study_substack_energy_dict_realtime[f'{cut_before},{cut_after}'][_substack_key]
        _substacks[f'{cut_before},{cut_after}'] = _substack_key
    case_study_best_combination_latency_realtime[','.join(str(i) for i in df_cut)] = {
        'latency': _latency,
        'energy' : _energy,
        'strategy' : _substacks
    }

case_study_best_combination_latency_realtime = dict(sorted(case_study_best_combination_latency_realtime.items(), key=lambda x: x[1]['latency']))

case_study_best_combination_energy_realtime: dict = {}
for df_cut in cut_id_candidate_list_new_hw:
    _latency = 0
    _energy = 0
    _substacks = {}
    for cut_before, cut_after in zip([None] + df_cut, df_cut + [_MAX]):
        _substack_key = list(case_study_substack_energy_dict_realtime[f'{cut_before},{cut_after}'].keys())[0]
        _latency += case_study_substack_latency_dict_realtime[f'{cut_before},{cut_after}'][_substack_key]
        _energy += case_study_substack_energy_dict_realtime[f'{cut_before},{cut_after}'][_substack_key]
        _substacks[f'{cut_before},{cut_after}'] = _substack_key
    case_study_best_combination_energy_realtime[','.join(str(i) for i in df_cut)] = {
        'latency': _latency,
        'energy' : _energy,
        'strategy' : _substacks
    }

case_study_best_combination_energy_realtime = dict(sorted(case_study_best_combination_energy_realtime.items(), key=lambda x: x[1]['energy']))

#%%
assert 0, "This is the end of world. Please refer to the final sorted dict for the DSE winner."