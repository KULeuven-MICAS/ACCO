import math
import pickle, os, time, re
import json
import itertools
import numpy as np

from classes.stages.MainInputParserStages import parse_workload_from_path
from classes.workload.layer_node import LayerNode, InputLayerNode
from classes.depthfirst.data_copy_layer import DataCopyLayer
from classes.depthfirst.data_copy_layer import DataCopyLayer

def extract_substack_readings(cm_list, layer, last_layer_idx, cut_range, data_movement_overhead, operand_max_hierarchy):
    _offload_latency_delta = 0
    _offload_energy_delta = 0
    _reload_latency_delta = 0
    _reload_energy_delta = 0
    _latency_2 = 0
    _energy_2 = 0
    _offloaded_data_bytes = 0
    _reloaded_data_bytes = 0
    for cm, mul in cm_list:
        _latency_2 += cm[0].latency_total2 * mul
        _energy_2 += cm[0].energy_total * mul

        # gather extra overhead for substack, in case of 'fake last' layer's output can be held on chip LLC
        if isinstance(layer, LayerNode) and layer.id != last_layer_idx and layer.id == cut_range[1]:
            _data_movement_bits_1 = cm[0].data_offloading_cc_pair_combined[2][0] * data_movement_overhead[1][2] * mul
            _offload_latency_delta += (cm[0].data_offloading_cycle - cm[0].data_offloading_cc_pair_combined[0][0] - cm[0].data_offloading_cc_pair_combined[1][0]) * mul

            assert (len(cm[0].energy_breakdown_further['O']) == operand_max_hierarchy[2]), "Substack Last Layer's Output ML does not have DRAM."
            _read_to_high = cm[0].energy_breakdown_further['O'][-2]['rd_out_to_high'] * mul
            _write_by_low = cm[0].energy_breakdown_further['O'][-1]['wr_in_by_low'] * mul
            assert (math.isclose(_write_by_low / data_movement_overhead[1][1] * data_movement_overhead[1][2], _read_to_high / data_movement_overhead[0][0] * data_movement_overhead[0][2], rel_tol=1e-3)), \
                f'Unmatched Energy Bits {layer.__str__()} write_by_low: {_write_by_low / data_movement_overhead[1][1] * data_movement_overhead[1][2]} <=> read_to_high: {_read_to_high / data_movement_overhead[0][0] * data_movement_overhead[0][2]}.'
            _data_movement_bits_2 = _write_by_low / data_movement_overhead[1][1] * data_movement_overhead[1][2]

            # assert(math.isclose(_data_movement_bits_1, _data_movement_bits_2, rel_tol=1e-3)), f'Unmatched {layer.__str__()} data_copy_action bits: {_data_movement_bits_1} <=> energy_breakdown bits: {_data_movement_bits_2}.'
            _offload_energy_delta += _read_to_high + _write_by_low
            _offloaded_data_bytes += _data_movement_bits_2 / 8.0

    # gather extra overhead for substack, in case of 'previous' layer's output can be found on chip LLC
    if isinstance(layer, DataCopyLayer) and 'substack_input_layer' in layer.__str__():
        _all_goes_to_GB = True
        for cm, mul in cm_list:
            if not _all_goes_to_GB:
                break
            for data_copy_action in cm[0].data_copy_actions:
                if len(data_copy_action.latency_breakdown) == 0:
                    assert (data_copy_action.data_amount > operand_max_hierarchy[3] * 8.0 * 1024 **2), f'data_copy_action amount: {data_copy_action.data_amount} not big enough'
                    _all_goes_to_GB = False

        if _all_goes_to_GB:
            _deeper_substack_copy = False
            for cm, mul in cm_list:
                _data_movement_bits_1 = 0
                _data_movement_bits_2 = 0
                for data_copy_action in cm[0].data_copy_actions:
                    _data_movement_bits_1 += data_copy_action.data_amount * mul
                    _reload_latency_delta += data_copy_action.latency_breakdown[0] * mul
                    if len(data_copy_action.latency_breakdown) > 1:
                        _deeper_substack_copy = True

                    assert (len(cm[0].energy_breakdown_further['I1']) == operand_max_hierarchy[0]), "Substack Input Layer's Input ML does not have DRAM."
                    _read_to_low = data_copy_action.energy_breakdown_further['I1'][-1]['rd_out_to_low'] * mul
                    _write_by_high = data_copy_action.energy_breakdown_further['I1'][-2]['wr_in_by_high'] * mul
                    assert (math.isclose(_read_to_low / data_movement_overhead[1][0] * data_movement_overhead[1][2], _write_by_high / data_movement_overhead[0][1] * data_movement_overhead[0][2], rel_tol=1e-3)), \
                        f'Unmatched Energy Bits {layer.__str__()} read_to_low: {_read_to_low / data_movement_overhead[1][0] * data_movement_overhead[1][2]} <=> write_by_high: {_write_by_high / data_movement_overhead[0][1] * data_movement_overhead[0][2]}.'
                    _data_movement_bits_2 += _read_to_low / data_movement_overhead[1][0] * data_movement_overhead[1][2]
                    _reload_energy_delta += _read_to_low + _write_by_high

                assert (
                    math.isclose(_data_movement_bits_1, _data_movement_bits_2, rel_tol=1e-3)), f'Unmatched {layer.__str__()} data_copy_action bits: {_data_movement_bits_1} <=> energy_breakdown bits: {_data_movement_bits_2}.'
                _reloaded_data_bytes += _data_movement_bits_2 / 8.0

            if _deeper_substack_copy:
                print(f'Layer: {layer.__str__()} of cut_range [{cut_range}] has a deeper-than-GB substack input loading.')

    return int(_latency_2), int(_energy_2), int(_offload_latency_delta), int(_offload_energy_delta), int(_reload_latency_delta), int(_reload_energy_delta), int(_reloaded_data_bytes), int(_offloaded_data_bytes)

def result_generator(
        root_dir, home_dir, workload, df_tilesize_x, df_tilesize_y, df_horizontal_caching, df_vertical_caching, df_tcn_global_initial_dilation, df_tcn_frame_amount,
        data_movement_overhead, operand_max_hierarchy, total_layers_dict, home_dir_list,
        _t_0
):
    _t_1 = time.time()

    var_names = locals()

    total_layers = total_layers_dict[workload.split('_')[1]]
    last_layer_idx = total_layers - 1
    path_format = f'{workload}_dil-{df_tcn_global_initial_dilation}_fr-{df_tcn_frame_amount}'
    file_format = f'{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}'

    log_dir = home_dir.replace(root_dir, root_dir+'_log')
    if not os.path.exists(f'{log_dir}/{path_format}'):
        os.makedirs(f'{log_dir}/{path_format}')
    else:
        print(f'\tRemoving previous logs: {workload} -> {file_format}')
        for fname in os.listdir(f'{log_dir}/{path_format}'):
            if re.match(r'FILE_.*\.log'.replace('FILE', file_format), fname):
                print(f'\t\tRemoving: {fname}')
                os.remove(f'{log_dir}/{path_format}/{fname}')

    var_names[f'{workload}_weight_cut_cases'] = []
    var_names[f'{workload}_cut_range_cases'] = {}
    with open(f'{home_dir}/{path_format}/{file_format}_[].pkl', 'rb') as fp:
        cme_ensemble = pickle.load(fp)
        for cme in cme_ensemble:
            if cme[1] not in var_names[f'{workload}_weight_cut_cases']:
                var_names[f'{workload}_weight_cut_cases'].append(cme[1])

            # if cme[2][0]['cut_before'] == 5 and cme[2][0]['cut_after'] == 8:
            #     cme_probe.append(cme)

            _cut_range = (cme[2][0]['cut_before'], cme[2][0]['cut_after'])
            if _cut_range not in var_names[f'{workload}_cut_range_cases']:
                var_names[f'{workload}_cut_range_cases'][_cut_range] = {
                    'energy_list_stable': [],
                    'latency_list_stable': [],
                    'offload_energy_delta_list_stable': [],
                    'offload_latency_delta_list_stable': [],
                    'reload_energy_delta_list_stable': [],
                    'reload_latency_delta_list_stable': [],
                    'data_offload_list_stable': [],
                    'data_reload_list_stable': [],

                    'energy_list_rampup': [],
                    'latency_list_rampup': [],
                    'offload_energy_delta_list_rampup': [],
                    'offload_latency_delta_list_rampup': [],
                    'reload_energy_delta_list_rampup': [],
                    'reload_latency_delta_list_rampup': [],
                    'data_offload_list_rampup': [],
                    'data_reload_list_rampup': [],
                    'layer_name_list': [],

                    'total_energy_per_layer': [],
                    'total_latency_per_layer': [],
                    'total_offload_energy_delta_per_layer': [],
                    'total_offload_latency_delta_per_layer': [],
                    'total_reload_energy_delta_per_layer': [],
                    'total_reload_latency_delta_per_layer': [],
                    'total_data_offload_per_layer': [],
                    'total_data_reload_per_layer': [],
                }

            layer = cme[0].layer
            # if layer.id == last_layer_idx:
            #     print(f'Last layer found: {layer.__str__()} @ range {_cut_range}')

            cm_list = cme[2][1]  # cost_model_evaluations_per_layer
            _latency, _energy, _offload_latency_delta, _offload_energy_delta, _reload_latency_delta, _reload_energy_delta, _reloaded_data_bytes, _offloaded_data_bytes = extract_substack_readings(cm_list, layer, last_layer_idx, _cut_range, data_movement_overhead, operand_max_hierarchy)  # _latency_1, _energy_1, _latency_2, _energy_2
            
            var_names[f'{workload}_cut_range_cases'][_cut_range]['total_energy_per_layer'].append(_energy)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['total_latency_per_layer'].append(_latency)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['total_offload_energy_delta_per_layer'].append(_offload_energy_delta)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['total_offload_latency_delta_per_layer'].append(_offload_latency_delta)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['total_reload_energy_delta_per_layer'].append(_reload_energy_delta)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['total_reload_latency_delta_per_layer'].append(_reload_latency_delta)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['total_data_offload_per_layer'].append(_offloaded_data_bytes)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['total_data_reload_per_layer'].append(_reloaded_data_bytes)

            cm_list = cme[2][2]  # cost_model_evaluations_per_layer_row_beginning
            _latency_0, _energy_0, _offload_latency_delta_0, _offload_energy_delta_0, _reload_latency_delta_0, _reload_energy_delta_0, _reloaded_data_bytes_0, _offloaded_data_bytes_0 = extract_substack_readings(cm_list, layer, last_layer_idx, _cut_range, data_movement_overhead, operand_max_hierarchy)  # _latency_1, _energy_1, _latency_2, _energy_2

            var_names[f'{workload}_cut_range_cases'][_cut_range]['energy_list_rampup'].append(_energy_0)  # tile-average
            var_names[f'{workload}_cut_range_cases'][_cut_range]['latency_list_rampup'].append(_latency_0)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['offload_energy_delta_list_rampup'].append(_offload_energy_delta_0)  # tile-average
            var_names[f'{workload}_cut_range_cases'][_cut_range]['offload_latency_delta_list_rampup'].append(_offload_latency_delta_0)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['reload_energy_delta_list_rampup'].append(_reload_energy_delta_0)  # tile-average
            var_names[f'{workload}_cut_range_cases'][_cut_range]['reload_latency_delta_list_rampup'].append(_reload_latency_delta_0)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['data_offload_list_rampup'].append(_offloaded_data_bytes_0)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['data_reload_list_rampup'].append(_reloaded_data_bytes_0)
            var_names[f'{workload}_cut_range_cases'][_cut_range]['layer_name_list'].append(layer.__str__())

            cm_list = cme[2][3]  # cost_model_evaluations_per_layer_row_internal
            _latency, _energy, _offload_latency_delta, _offload_energy_delta, _reload_latency_delta, _reload_energy_delta, _reloaded_data_bytes, _offloaded_data_bytes = extract_substack_readings(cm_list, layer, last_layer_idx, _cut_range, data_movement_overhead, operand_max_hierarchy)  # _latency_1, _energy_1, _latency_2, _energy_2
            if len(cm_list) != 0:

                var_names[f'{workload}_cut_range_cases'][_cut_range]['energy_list_stable'].append(_energy)  # tile-average
                var_names[f'{workload}_cut_range_cases'][_cut_range]['latency_list_stable'].append(_latency)
                var_names[f'{workload}_cut_range_cases'][_cut_range]['offload_energy_delta_list_stable'].append(_offload_energy_delta)  # tile-average
                var_names[f'{workload}_cut_range_cases'][_cut_range]['offload_latency_delta_list_stable'].append(_offload_latency_delta)
                var_names[f'{workload}_cut_range_cases'][_cut_range]['reload_energy_delta_list_stable'].append(_reload_energy_delta)  # tile-average
                var_names[f'{workload}_cut_range_cases'][_cut_range]['reload_latency_delta_list_stable'].append(_reload_latency_delta)
                var_names[f'{workload}_cut_range_cases'][_cut_range]['data_offload_list_stable'].append(_offloaded_data_bytes)
                var_names[f'{workload}_cut_range_cases'][_cut_range]['data_reload_list_stable'].append(_reloaded_data_bytes)

            # no such stable scenario as the tile-size is too large;
            # fall back to the rampup;
            else:

                var_names[f'{workload}_cut_range_cases'][_cut_range]['energy_list_stable'].append(_energy_0)  # tile-average
                var_names[f'{workload}_cut_range_cases'][_cut_range]['latency_list_stable'].append(_latency_0)
                var_names[f'{workload}_cut_range_cases'][_cut_range]['offload_energy_delta_list_stable'].append(_offload_energy_delta_0)  # tile-average
                var_names[f'{workload}_cut_range_cases'][_cut_range]['offload_latency_delta_list_stable'].append(_offload_latency_delta_0)
                var_names[f'{workload}_cut_range_cases'][_cut_range]['reload_energy_delta_list_stable'].append(_reload_energy_delta_0)  # tile-average
                var_names[f'{workload}_cut_range_cases'][_cut_range]['reload_latency_delta_list_stable'].append(_reload_latency_delta_0)
                var_names[f'{workload}_cut_range_cases'][_cut_range]['data_offload_list_stable'].append(_offloaded_data_bytes_0)
                var_names[f'{workload}_cut_range_cases'][_cut_range]['data_reload_list_stable'].append(_reloaded_data_bytes_0)

        del cme_ensemble

    for _cut_range in var_names[f'{workload}_cut_range_cases'].keys():
        # hooks to reuse some old file writers
        energy_list_stable = var_names[f'{workload}_cut_range_cases'][_cut_range]['energy_list_stable']
        latency_list_stable = var_names[f'{workload}_cut_range_cases'][_cut_range]['latency_list_stable']
        offload_energy_delta_list_stable = var_names[f'{workload}_cut_range_cases'][_cut_range]['offload_energy_delta_list_stable']
        offload_latency_delta_list_stable = var_names[f'{workload}_cut_range_cases'][_cut_range]['offload_latency_delta_list_stable']
        reload_energy_delta_list_stable = var_names[f'{workload}_cut_range_cases'][_cut_range]['reload_energy_delta_list_stable']
        reload_latency_delta_list_stable = var_names[f'{workload}_cut_range_cases'][_cut_range]['reload_latency_delta_list_stable']

        energy_list_rampup = var_names[f'{workload}_cut_range_cases'][_cut_range]['energy_list_rampup']
        latency_list_rampup = var_names[f'{workload}_cut_range_cases'][_cut_range]['latency_list_rampup']
        offload_energy_delta_list_rampup = var_names[f'{workload}_cut_range_cases'][_cut_range]['offload_energy_delta_list_rampup']
        offload_latency_delta_list_rampup = var_names[f'{workload}_cut_range_cases'][_cut_range]['offload_latency_delta_list_rampup']
        reload_energy_delta_list_rampup = var_names[f'{workload}_cut_range_cases'][_cut_range]['reload_energy_delta_list_rampup']
        reload_latency_delta_list_rampup = var_names[f'{workload}_cut_range_cases'][_cut_range]['reload_latency_delta_list_rampup']

        layer_name_list = var_names[f'{workload}_cut_range_cases'][_cut_range]['layer_name_list']
        total_energy_per_layer = var_names[f'{workload}_cut_range_cases'][_cut_range]['total_energy_per_layer']
        total_latency_per_layer = var_names[f'{workload}_cut_range_cases'][_cut_range]['total_latency_per_layer']
        total_offload_energy_delta_per_layer = var_names[f'{workload}_cut_range_cases'][_cut_range]['total_offload_energy_delta_per_layer']
        total_offload_latency_delta_per_layer = var_names[f'{workload}_cut_range_cases'][_cut_range]['total_offload_latency_delta_per_layer']
        total_reload_energy_delta_per_layer = var_names[f'{workload}_cut_range_cases'][_cut_range]['total_reload_energy_delta_per_layer']
        total_reload_latency_delta_per_layer = var_names[f'{workload}_cut_range_cases'][_cut_range]['total_reload_latency_delta_per_layer']

        total_data_offload_per_layer = var_names[f'{workload}_cut_range_cases'][_cut_range]['total_data_offload_per_layer']
        data_offload_list_rampup = var_names[f'{workload}_cut_range_cases'][_cut_range]['data_offload_list_rampup']
        data_offload_list_stable = var_names[f'{workload}_cut_range_cases'][_cut_range]['data_offload_list_stable']

        total_data_reload_per_layer = var_names[f'{workload}_cut_range_cases'][_cut_range]['total_data_reload_per_layer']
        data_reload_list_rampup = var_names[f'{workload}_cut_range_cases'][_cut_range]['data_reload_list_rampup']
        data_reload_list_stable = var_names[f'{workload}_cut_range_cases'][_cut_range]['data_reload_list_stable']

        cut_before = _cut_range[0]
        cut_after = _cut_range[1]

        with open(
                f'{log_dir}/{path_format}/{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}_frame_summary.log',
                'a+') as f:
            string_lat = ','.join(str(i) for i in total_latency_per_layer)
            string_ene = ','.join(str(i) for i in total_energy_per_layer)
            string_offload_lat_delta = ','.join(str(i) for i in total_offload_latency_delta_per_layer)
            string_offload_ene_delta = ','.join(str(i) for i in total_offload_energy_delta_per_layer)
            string_reload_lat_delta = ','.join(str(i) for i in total_reload_latency_delta_per_layer)
            string_reload_ene_delta = ','.join(str(i) for i in total_reload_energy_delta_per_layer)
            string_data_offload = ','.join(str(i) for i in total_data_offload_per_layer)
            string_data_reload = ','.join(str(i) for i in total_data_reload_per_layer)
            string_layer_name = ','.join(str(i) for i in layer_name_list)
            f.write(f'{cut_before},{cut_after},{string_lat}\n')
            f.write(f'{cut_before},{cut_after},{string_offload_lat_delta}\n')
            f.write(f'{cut_before},{cut_after},{string_reload_lat_delta}\n')
            f.write(f'{cut_before},{cut_after},{string_ene}\n')
            f.write(f'{cut_before},{cut_after},{string_offload_ene_delta}\n')
            f.write(f'{cut_before},{cut_after},{string_reload_ene_delta}\n')
            f.write(f'{cut_before},{cut_after},{string_data_offload}\n')
            f.write(f'{cut_before},{cut_after},{string_data_reload}\n')
            f.write(f'{cut_before},{cut_after},{string_layer_name}\n')

        with open(
                f'{log_dir}/{path_format}/{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}_frame_total.log',
                'a+') as f:
            f.write(f'{cut_before},{cut_after},')
            f.write(f'{int(sum(total_latency_per_layer))},')
            f.write(f'{int(sum(total_energy_per_layer))},')
            f.write(f'{int(sum(total_offload_latency_delta_per_layer))},')
            f.write(f'{int(sum(total_offload_energy_delta_per_layer))},')
            f.write(f'{int(sum(total_reload_latency_delta_per_layer))},')
            f.write(f'{int(sum(total_reload_energy_delta_per_layer))},')
            f.write(f'{int(sum(total_data_offload_per_layer))},')
            f.write(f'{int(sum(total_data_reload_per_layer))},\n')

        with open(
                f'{log_dir}/{path_format}/{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}.log',
                'a+') as f:
            f.write(f'[{cut_before},{cut_after}]-energy_per_layer_stable: {energy_list_stable}\n')
            f.write(f'[{cut_before},{cut_after}]-offload_energy_delta_per_layer_stable: {offload_energy_delta_list_stable}\n')
            f.write(f'[{cut_before},{cut_after}]-reload_energy_delta_per_layer_stable: {reload_energy_delta_list_stable}\n')
            f.write(f'[{cut_before},{cut_after}]-latency_per_layer_stable: {latency_list_stable}\n')
            f.write(f'[{cut_before},{cut_after}]-offload_latency_delta_per_layer_stable: {offload_latency_delta_list_stable}\n')
            f.write(f'[{cut_before},{cut_after}]-reload_latency_delta_per_layer_stable: {reload_latency_delta_list_stable}\n')
            f.write(f'[{cut_before},{cut_after}]-energy_per_layer_rampup: {energy_list_rampup}\n')
            f.write(f'[{cut_before},{cut_after}]-offload_energy_delta_per_layer_rampup: {offload_energy_delta_list_rampup}\n')
            f.write(f'[{cut_before},{cut_after}]-reload_energy_delta_per_layer_rampup: {reload_energy_delta_list_rampup}\n')
            f.write(f'[{cut_before},{cut_after}]-latency_per_layer_rampup: {latency_list_rampup}\n')
            f.write(f'[{cut_before},{cut_after}]-offload_latency_delta_per_layer_rampup: {offload_latency_delta_list_rampup}\n')
            f.write(f'[{cut_before},{cut_after}]-reload_latency_delta_per_layer_rampup: {reload_latency_delta_list_rampup}\n')
            f.write(f'[{cut_before},{cut_after}]-data_offload_list_stable: {data_offload_list_stable}\n')
            f.write(f'[{cut_before},{cut_after}]-data_offload_list_rampup: {data_offload_list_rampup}\n')
            f.write(f'[{cut_before},{cut_after}]-data_reload_list_stable: {data_offload_list_stable}\n')
            f.write(f'[{cut_before},{cut_after}]-data_reload_list_rampup: {data_offload_list_rampup}\n')

        with open(
                f'{log_dir}/{path_format}/{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}_layer_summary.log',
                'a+') as f:
            energy_stable = sum(energy_list_stable)
            latency_stable = sum(latency_list_stable)
            offload_energy_stable = sum(offload_energy_delta_list_stable)
            offload_latency_stable = sum(offload_latency_delta_list_stable)
            reload_energy_stable = sum(reload_energy_delta_list_stable)
            reload_latency_stable = sum(reload_latency_delta_list_stable)

            energy_rampup = sum(energy_list_rampup)
            latency_rampup = sum(latency_list_rampup)
            offload_energy_rampup = sum(offload_energy_delta_list_rampup)
            offload_latency_rampup = sum(offload_latency_delta_list_rampup)
            reload_energy_rampup = sum(reload_energy_delta_list_rampup)
            reload_latency_rampup = sum(reload_latency_delta_list_rampup)

            data_offload_stable = sum(data_offload_list_stable)
            data_offload_rampup = sum(data_offload_list_rampup)
            data_reload_stable = sum(data_reload_list_stable)
            data_reload_rampup = sum(data_reload_list_rampup)

            f.write(
                f'{cut_before},{cut_after},'
                f'{energy_stable},{latency_stable},{offload_energy_stable},{offload_latency_stable},{reload_energy_stable},{reload_latency_stable},'
                f'{energy_rampup},{latency_rampup},{offload_energy_rampup},{offload_latency_rampup},{reload_energy_rampup},{reload_latency_rampup},'
                f'{data_offload_stable},{data_offload_rampup},'
                f'{data_reload_stable},{data_reload_rampup}\n')

        with open(
                f'{log_dir}/{path_format}/{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}_helper.log',
                'a+') as f:
            f.write(f'[{cut_before},{cut_after}]-layer_name_list: \n{layer_name_list}\n')

    del var_names[f'{workload}_cut_range_cases']

    _t_2 = time.time()
    print(f'Working on: home_dir[{home_dir_list.index(home_dir)}] -> {workload} -> {file_format}')
    print(f'Time Elapsed: {round(_t_2 - _t_1, 2)} s / {round(_t_2 - _t_0, 2)} s')