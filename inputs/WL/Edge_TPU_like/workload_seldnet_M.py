workload = {
    -1: {'equation': 'input',
         'loop_dim_size': {'B': 1, 'K': 3, 'OY': 224, 'OX': 224},
         'precision': 8,
         'core_allocation': 1,
         'memory_operand_links': {'O': 'I1'}
    }
    ,
    0: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 128, 'C': 16, 'OY': 64, 'OX': 512, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [-1]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 8)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    1: {  # max pool / s4
        'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
        'equation_relations': ['ix=4*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'G': 128, 'OY': 64, 'OX': 128, 'FX': 4, 'FY': 1},
        'operand_precision': {'O': 16, 'O_final': 8, 'I': 8, 'W': 0},
        'operand_source': {'W': [], 'I': [0]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('G', 8)},
        'memory_operand_links': {'O': 'O', 'I': 'I1', 'W': 'I2'}
    }
    ,
    2: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 128, 'C': 128, 'OY': 64, 'OX': 512, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 8)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    3: {  # max pool / s4
        'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
        'equation_relations': ['ix=4*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'G': 128, 'OY': 64, 'OX': 32, 'FX': 4, 'FY': 1},
        'operand_precision': {'O': 16, 'O_final': 8, 'I': 8, 'W': 0},
        'operand_source': {'W': [], 'I': [2]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('G', 8)},
        'memory_operand_links': {'O': 'O', 'I': 'I1', 'W': 'I2'}
    }
    ,
    4: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 128, 'C': 128, 'OY': 64, 'OX': 32, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [3]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 8)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    5: {  # max pool / s4
        'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
        'equation_relations': ['ix=4*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'G': 128, 'OY': 64, 'OX': 8, 'FX': 4, 'FY': 1},
        'operand_precision': {'O': 16, 'O_final': 8, 'I': 8, 'W': 0},
        'operand_source': {'W': [], 'I': [4]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('G', 8)},
        'memory_operand_links': {'O': 'O', 'I': 'I1', 'W': 'I2'}
    }
    ,
    6: {  # Conv / s1 / T * 0.5M * 2C for P filters in [16 ~ 512], T in [32~512], M=1024, C=8
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 128, 'C': 128, 'OY': 64, 'OX': 8, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [5]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 8)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    7: {  # max pool / s4
        'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
        'equation_relations': ['ix=4*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'G': 128, 'OY': 64, 'OX': 2, 'FX': 4, 'FY': 1},
        'operand_precision': {'O': 16, 'O_final': 8, 'I': 8, 'W': 0},
        'operand_source': {'W': [], 'I': [6]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('G', 8)},
        'memory_operand_links': {'O': 'O', 'I': 'I1', 'W': 'I2'}
    }
}
