workload = {
    -2: {  # input features, 1 frame of spectrogram, 128*64
        'equation': 'input',
        'loop_dim_size': {'B': 1, 'K': 3, 'OY': 130, 'OX': 127},  # OX=100, frame width 64, 63+1=64 frames
        'precision': 8,
        'core_allocation': 1,
        'memory_operand_links': {'O': 'I1'}
    }
    ,
    -1: {  # test of sliding-window input sampling
        # original
        # 'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        # 'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        # 'loop_dim_size': {'B': 1, 'K': 64, 'C': 3, 'OY': 311, 'OX': 311, 'FY': 3, 'FX': 3},
        # 'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        # 'operand_source': {'W': [], 'I': [-1]},
        # 'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        # 'constant_operands': ['W'],
        # 'core_allocation': 1,
        # 'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        # 'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}

        # # ref to resnet residual definition form
        'equation': 'O[nb][k][oy][ox]=I[b][k][ix][iy]',
        'equation_relations': ['ix=1*ox+1*nb', 'iy=1*oy+0*nb'],
        'loop_dim_size': {'B': 1, 'NB': 64, 'K': 3, 'OY': 130, 'OX': 64},
        'operand_precision': {'O': 8, 'O_final': 8, 'I': 8},
        'operand_source': {'I': [-2]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'K': 'K'}},
        # 'constant_operands': [],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('NB', 32), 'D2': ('K', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'I': 'I1'}

        # # ref to resnet residual definition form + MaxPooling W implementation (stride-1 pooling)
        # 'equation': 'O[nb][k][oy][ox]=W[fx][fy]*I[b][k][ix][iy]',
        # 'equation_relations': ['ix=1*ox+1*nb', 'iy=1*oy+0*nb'],
        # 'loop_dim_size': {'B': 1, 'NB': 64, 'K': 3, 'OY': 130, 'OX': 64, 'FY': 1, 'FX': 1},
        # 'operand_precision': {'O': 8, 'O_final': 8, 'I': 8, 'W': 0},
        # 'operand_source': {'I': [-2]},
        # 'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'K': 'K'}},
        # # 'constant_operands': [],
        # 'core_allocation': 1,
        # 'spatial_mapping': {'D1': ('NB', 32), 'D2': ('K', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        # 'memory_operand_links': {'O': 'O', 'I': 'I1', 'W': 'I2'}
    },
    0: {  # conv1, stride 1
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 64, 'C': 3, 'OY': 311, 'OX': 311, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [-1]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    # ,
    # 1: {  # max pool, stride 2
    #     'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
    #     'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
    #     'loop_dim_size': {'B': 1, 'G': 64, 'OY': 155, 'OX': 155, 'FX': 3, 'FY': 3},
    #     'operand_precision': {'O': 16, 'O_final': 8, 'I': 8, 'W': 0},
    #     'operand_source': {'W': [], 'I': [0]},
    #     'constant_operands': ['W'],
    #     'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
    #     'core_allocation': 1,
    #     'spatial_mapping': {'D1': ('G', 32), 'D3': ('OX', 4), 'D4': ('OY', 4)},
    #     'memory_operand_links': {'O': 'O', 'I': 'I1', 'W': 'I2'}
    # }
    # ,
    # 3: {  # fc
    #     'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
    #     'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
    #     'loop_dim_size': {'B': 1, 'K': 1000, 'C': 512, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1},
    #     'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
    #     'operand_source': {'W': [], 'I': [26]},
    #     'constant_operands': ['W'],
    #     'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'G'}},
    #     'core_allocation': 1,
    #     'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2)},
    #     'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    # }
}
