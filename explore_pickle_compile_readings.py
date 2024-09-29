#%%
import pickle, os, time
import json
import itertools
import multiprocessing as mp
import numpy as np

from classes.stages.MainInputParserStages import parse_workload_from_path
from classes.workload.layer_node import LayerNode, InputLayerNode
from classes.depthfirst.data_copy_layer import DataCopyLayer

#%%

# location of the DSE cost model pkl files
root_dir = './result_pickle_files_misc_camready'
batch_dir_list = ['test_Edge_TPU_like', 'test_Meta_prototype', 'test_Tesla_NPU_like']

workload_list = [
    # 'workload_stftCNN_fixed_unrolling_oxoy_025',
    # 'workload_stftCNN_fixed_unrolling_oxoy_050',
    # 'workload_stftCNN_fixed_unrolling_oxoy_075',
    # 'workload_stftCNN_fixed_unrolling_oxoy_100',
    # 'workload_resnet18_corrected_fixed_unrolling_oxoy',
    # 'workload_mobilenetv1_corrected_fixed_unrolling_oxoy',
    'workload_seldnet_oxoy_M',
    # 'workload_stftCNN_fixed_unrolling_oxoy_025_1x',
    # 'workload_stftCNN_fixed_unrolling_oxoy_025_16x'
]

# workload basic info
total_layers_dict = {
    'seldnet': 8,
    'stftCNN': 12,
    'mobilenetv1': 27,
    'resnet18': 26
}

# some green sheets of the HW model
# LLC writing / DRAM writing size and cost
operand_max_hierarchy_dict = { # I1-level, I2-level, O-level, GB-IO-sizeMB, GB-W-sizeMB
    'test_Edge_TPU_like': [3, 4, 4, 1.0, 1.0],
    'test_Meta_prototype': [3, 4, 4, 1.0, 1.0],
    'test_Tesla_NPU_like': [4, 5, 4, 896/1024.0, 1.0]
}
data_movement_overhead_dict = { # [read-energy/cc, write-energy/cc, bw/cc]
    'test_Edge_TPU_like': [[26.01 * 8, 23.65 * 8, 128 * 8], [700 * 1, 750 * 1, 64 * 1]],
    'test_Meta_prototype': [[26.01 * 8, 23.65 * 8, 128 * 8], [700 * 1, 750 * 1, 64 * 1]],
    'test_Tesla_NPU_like': [[26.01 * 7, 23.65 * 8, 128 * 7], [700 * 1, 750 * 1, 64 * 1]] 
}
#%%
# specify which scenarios to retrieve and analyze

opt_strategy_list = ['energy', 'latency']
df_tcn_global_initial_dilation_list = [-1]
df_tcn_frame_amount = 1

df_tcn_global_initial_dilation_list = [-1]
df_x_tilesizes = [1, 2, 4, 8, 16, 32, 64, 128]
df_y_tilesizes = [1, 2, 4, 8, 16, 32, 64]

# df_caching_list = [[False, False], [True, False], [False, True], [False, False]]
df_caching_list = [[False, False]]

var_names = locals()

home_dir_list = []
for batch_dir, opt_strategy in itertools.product(batch_dir_list, opt_strategy_list):
    home_dir_list.append(f'{root_dir}/{batch_dir}/test_mp_{opt_strategy}')

from explore_pickle_compile_readings_utils import *
#%%
# parse the .pkl cost model ensembles and generate reports
max_multi_processing = 16
multi_threading_amount = min(max_multi_processing, mp.cpu_count())
mp_pool = mp.Pool(multi_threading_amount)

_t_0 = time.time()
_t_0_formatted = time.strftime("%y%m%d-%H%M%S", time.localtime())
print(f'Jobs Started @ {multi_threading_amount}c: {_t_0_formatted}')

for home_dir, workload, df_tilesize_x, df_tilesize_y, [df_horizontal_caching, df_vertical_caching], df_tcn_global_initial_dilation in \
        itertools.product(
            home_dir_list,
            workload_list,
            df_x_tilesizes,
            df_y_tilesizes,
            df_caching_list,
            df_tcn_global_initial_dilation_list
        ):

    # get the hw name to look up
    for folder_name in home_dir.split('/'):
        if folder_name in batch_dir_list:
            hw_scenario = folder_name
    data_movement_overhead = data_movement_overhead_dict[hw_scenario]
    operand_max_hierarchy = operand_max_hierarchy_dict[hw_scenario]
    
    mp_pool.apply_async(result_generator, args=(
        root_dir, home_dir, workload, df_tilesize_x, df_tilesize_y, df_horizontal_caching, df_vertical_caching, df_tcn_global_initial_dilation, df_tcn_frame_amount,
        data_movement_overhead, operand_max_hierarchy, total_layers_dict, home_dir_list,
        _t_0
    ))

mp_pool.close()
mp_pool.join()

_t_2 = time.time()
print(f'Jobs Finished: {time.strftime("%y%m%d-%H%M%S", time.localtime())}')
print(f'Total Execution Time: {round(_t_2 - _t_0, 2)} seconds.')