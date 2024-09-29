workload = {
    -1: {'equation': 'input',
         'loop_dim_size': {'B': 1, 'K': 1, 'OY': 128, 'OX': 64},  # frame width 64, 64-1=63 frames
         'precision': 8,
         'core_allocation': 1,
         'memory_operand_links': {'O': 'I1'}
         }
    ,
    0: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 48, 'C': 1, 'OY': 128, 'OX': 128, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [-1]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    1: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 48, 'C': 48, 'OY': 128, 'OX': 128, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    2: {  # max pool / s2
        'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
        'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'G': 48, 'OY': 48, 'OX': 64, 'FX': 2, 'FY': 2},
        'operand_precision': {'O': 16, 'O_final': 8, 'I': 8, 'W': 0},
        'operand_source': {'W': [], 'I': [1]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('G', 32), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'I': 'I1', 'W': 'I2'}
    },
    3: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 96, 'C': 48, 'OY': 64, 'OX': 64, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    4: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 96, 'C': 96, 'OY': 64, 'OX': 64, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [3]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    5: {  # max pool / s2
        'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
        'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'G': 96, 'OY': 32, 'OX': 32, 'FX': 2, 'FY': 2},
        'operand_precision': {'O': 16, 'O_final': 8, 'I': 8, 'W': 0},
        'operand_source': {'W': [], 'I': [4]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('G', 32), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'I': 'I1', 'W': 'I2'}
    }
    ,
    6: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 192, 'C': 96, 'OY': 32, 'OX': 32, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [5]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    7: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 192, 'C': 192, 'OY': 32, 'OX': 32, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [6]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    8: {  # max pool / s2
        'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
        'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'G': 192, 'OY': 16, 'OX': 16, 'FX': 2, 'FY': 2},
        'operand_precision': {'O': 16, 'O_final': 8, 'I': 8, 'W': 0},
        'operand_source': {'W': [], 'I': [7]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('G', 32), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'I': 'I1', 'W': 'I2'}
    }
    ,
    9: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 384, 'C': 192, 'OY': 16, 'OX': 16, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [8]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    10: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 384, 'C': 384, 'OY': 16, 'OX': 16, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [9]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    11: {  # max pool / s2
        'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
        'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'G': 384, 'OY': 1, 'OX': 1, 'FX': 2, 'FY': 2},
        'operand_precision': {'O': 16, 'O_final': 8, 'I': 8, 'W': 0},
        'operand_source': {'W': [], 'I': [10]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('G', 32), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'I': 'I1', 'W': 'I2'}
    }
}
