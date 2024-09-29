import logging
import os.path

from typing import Generator, Callable, List, Tuple, Any

from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from classes.stages.Stage import Stage
from classes.cost_model.cost_model import CostModelEvaluation
import multiprocessing as mp
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)


class DfStackCutAndXform(Stage):
    def __init__(self, list_of_callables, *, accelerator, workload, df_stack_cuts:list, **kwargs):
        """
        Initialize the compare stage.
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.df_stack_cuts = df_stack_cuts

    def run(self):
        # print(self.df_stack_cuts)
        nodes = list(self.workload.topological_sort())


        if not self.df_stack_cuts:
            print("empty")
        else:
            print("not empty")

if __name__ == '__main__':
    class Dummy(Stage):
        def is_leaf(self):
            return True
        def run(self):
            yield None, self.kwargs


    _logging_level = logging.INFO
    _logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=_logging_level,
                         format=_logging_format)

    # from zigzag.classes.stages import SpatialMappingGeneratorStage
    from classes.stages.Stage import MainStage
    from classes.stages import \
        WorkloadAndAcceleratorParserStage, \
        DfStackCutIfWeightsOverflowStage, \
        DepthFirstStage, \
        SpatialMappingConversionStage, \
        RemoveExtraInfoStage, \
        MinimalEnergyStage, \
        MinimalLatencyStage, \
        ZigZagCostModelStage, \
        LomaStage, \
        GeneralParameterIteratorStage, \
        DumpStage, \
        SpatialMappingGeneratorStage
        # SalsaStage, \
    
    # import hardware models
    HW_WL_list = ['Meta_prototype', 'Edge_TPU_like', 'Tesla_NPU_like']

    # import target workload models
    # to run more models, please refer to the workload_folder/HW_WL
    workload_folder = 'inputs'
    workload_list = ['workload_seldnet_oxoy_M', 'workload_stftCNN_fixed_unrolling_oxoy_025_1x', 'workload_stftCNN_fixed_unrolling_oxoy_025_16x']

    # whether the depth-first caching is activated on x-axis or y-axis
    df_caching_list = [[False, False], [True, False], [False, True], [False, False]]
    
    # the layer-id for explicit depth-first substack cut
    df_cut_list = [[]]
    # the amount of further substack cuts, on top of the explicit df_cut_list, which is automatically explored by the tool
    df_tcn_substack_cut_max = 0
    # whether the substack's input layer is tiled, -1 means using the entire input as in the original workload
    df_tcn_global_initial_dilation_list = [-1]
    # batch amount, set to 1 by default
    df_tcn_frame_amount = 1
    
    # candidate x-axis and y-axis tiling sizes
    df_y_tilesizes = [1, 2, 4, 8, 16, 32, 64]
    df_x_tilesizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # layer-wise optimization target: min-energy or min-latency
    opt_strategy_list = ['energy', 'latency']

    mp.set_start_method('spawn')
    max_multi_threading = 32
    multi_threading_amount = min(max_multi_threading, mp.cpu_count())
    mp_pool = mp.Pool(multi_threading_amount)

    _t_1 = time.time()
    _t_1_formatted = time.strftime("%y%m%d-%H%M%S", time.localtime())
    print(f'Jobs Started @ {multi_threading_amount}c: {_t_1_formatted}')

    # spawn all the scenarios respectively
    for hw_wl_name in HW_WL_list:
        for opt_strategy in opt_strategy_list:
            for workload in workload_list:
                for df_tcn_global_initial_dilation in df_tcn_global_initial_dilation_list:
                    for df_tilesize_x in df_x_tilesizes:
                        for df_tilesize_y in df_y_tilesizes:
                            for df_cut in df_cut_list:
                                for [df_horizontal_caching, df_vertical_caching] in df_caching_list:
                                    result_suffix = f'{workload}_dil-{df_tcn_global_initial_dilation}_fr-{df_tcn_frame_amount}'
                                    result_saving_path = f'./result_pickle_files_misc_camready/test_{hw_wl_name}/test_mp_{opt_strategy}/{result_suffix}'

                                    if not os.path.exists(result_saving_path):
                                        os.makedirs(result_saving_path)

                                    _pkl_path = f'{result_saving_path}/{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}_{str(df_cut)}.pkl'
                                    if os.path.exists(_pkl_path):
                                        _ctime = int(time.strftime('%m%d%H', time.localtime(os.path.getmtime(_pkl_path))))
                                        if os.path.getsize(_pkl_path) <= 100:
                                            print(f'Remove: {result_saving_path}/{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}_{str(df_cut)}')
                                            os.remove(_pkl_path)
                                        else:
                                            print(f'Skip: {result_saving_path}/{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}_{str(df_cut)}')
                                            continue
                                    else:
                                        pass

                                    if opt_strategy == 'energy':
                                        optStage = MinimalEnergyStage
                                    else:
                                        optStage = MinimalLatencyStage

                                    DUT = MainStage(
                                        [
                                            WorkloadAndAcceleratorParserStage,
                                            DumpStage,
                                            DfStackCutIfWeightsOverflowStage,
                                            DepthFirstStage,
                                            SpatialMappingConversionStage,
                                            RemoveExtraInfoStage,
                                            optStage,
                                            LomaStage,
                                            ZigZagCostModelStage
                                        ],
                                        workload_path=f'{workload_folder}.WL.{hw_wl_name}.{workload}',
                                        accelerator_path=f'{workload_folder}.HW.{hw_wl_name}_DF',
                                        df_horizontal_caching=df_horizontal_caching,
                                        df_vertical_caching=df_vertical_caching,
                                        df_tilesize_x=df_tilesize_x,
                                        df_tilesize_y=df_tilesize_y,
                                        df_stack_cuts=df_cut,
                                        df_tcn_transform=True,
                                        df_tcn_global_initial_dilation=df_tcn_global_initial_dilation,
                                        df_tcn_frame_amount=df_tcn_frame_amount,
                                        df_tcn_substack_cut_max=df_tcn_substack_cut_max,
                                        loma_lpf_limit=6,
                                        result_saving_path= result_saving_path,
                                        dump_filename_pattern=f'{result_saving_path}/{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}_{str(df_cut)}.pkl',
                                    )

                                    DUT.run()

    mp_pool.close()
    mp_pool.join()

    _t_2 = time.time()
    print(f'Jobs Finished: {time.strftime("%y%m%d-%H%M%S", time.localtime())}')
    print(f'Total Execution Time: {round(_t_2 - _t_1, 3)} seconds.')