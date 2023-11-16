import numpy as np

# Traditional 4 element tetris blocks
tetris_I = np.array([[0, 0, 0, 0]])           # 1
tetris_J = np.array([[0, 1, 1],               # 2
                     [0, 0, 0]])
tetris_L = np.array([[1, 1, 0],               # 3
                     [0, 0, 0]])
tetris_O = np.array([[0, 0],                  # 4
                     [0, 0]])
tetris_S = np.array([[1, 0, 0],               # 5
                     [0, 0, 1]])
tetris_T = np.array([[1, 0, 1],               # 6
                     [0, 0, 0]])
tetris_Z = np.array([[0, 0, 1],               # 7
                     [1, 0, 0]])

# 5 element blocks used in Blokus
tetris_5p_1 = np.array([[0, 0, 0],            # 8
                        [0, 1, 0]])
tetris_5p_2 = np.array([[0, 0, 0, 0, 0]])     # 9
tetris_5p_3 = np.array([[0, 1, 1],            # 10
                        [0, 1, 1],
                        [0 ,0 ,0]])
tetris_5p_4 = np.array([[1, 0, 1],            # 11
                        [0, 0, 0],
                        [1, 0, 1]])
tetris_5p_5 = np.array([[0, 0, 0, 0],         # 12
                        [1, 1, 1, 0]])
tetris_5p_5r = np.array([[0, 0, 0, 0],        # 13
                         [0, 1, 1, 1]])
tetris_5p_6 = np.array([[0, 0, 0, 0],         # 14
                        [1, 1, 0, 1]])
tetris_5p_6r = np.array([[0, 0, 0, 0],        # 15
                         [1, 0, 1, 1]])
tetris_5p_7 = np.array([[1, 1, 0],            # 16
                        [0, 0, 0],
                        [0, 1, 1]])
tetris_5p_7r = np.array([[0, 1, 1],           # 17
                         [0, 0, 0],
                         [1, 1, 0]])
tetris_5p_8 = np.array([[1, 1, 0],            # 18
                        [1, 0, 0],
                        [0, 0, 1]])
tetris_5p_9 = np.array([[0, 0, 1],            # 19
                        [0, 0, 0]])
tetris_5p_9r = np.array([[0, 0, 0],           # 20
                         [0, 0, 1]])
tetris_5p_10 = np.array([[0, 1, 1],           # 21
                         [0, 0, 0],
                         [0, 1, 1]])
tetris_5p_11 = np.array([[0, 0, 1],           # 22
                         [1, 0, 0],
                         [1, 0, 1]])
tetris_5p_11r = np.array([[1, 0, 0],          # 23
                          [0, 0, 1],
                          [1, 0, 1]])
tetris_5p_12 = np.array([[1, 0],              # 24
                         [0, 0],
                         [0, 1],
                         [0, 1]])
tetris_5p_12r = np.array([[0, 1],             # 25
                          [0, 0],
                          [1, 0],
                          [1, 0]])

tetris_blocks = [tetris_I, tetris_J, tetris_L, tetris_O, tetris_S, tetris_T, tetris_Z]
blokus_blocks = [tetris_5p_1, tetris_5p_2, tetris_5p_3, tetris_5p_4,
                 tetris_5p_5, tetris_5p_5r, tetris_5p_6, tetris_5p_6r,
                 tetris_5p_7, tetris_5p_7r, tetris_5p_8, tetris_5p_9,
                 tetris_5p_9r, tetris_5p_10, tetris_5p_11, tetris_5p_11r,
                 tetris_5p_12, tetris_5p_12r]

total_blocks = tetris_blocks + blokus_blocks

block_colors =  {0 : np.array([255, 255, 255]),
                 1 : np.array([70, 240, 240]),
                 2 : np.array([0, 0, 240]),
                 3 : np.array([0, 0, 240]),
                 4 : np.array([240, 240, 0]),
                 5 : np.array([240, 0, 0]),
                 6 : np.array([160, 0, 240]),
                 7 : np.array([240, 0, 0]),
                 8 : np.array([128, 0, 0]),
                 9 : np.array([250, 190, 212]),
                 10: np.array([170, 110, 40]),
                 11: np.array([255, 215, 180]),
                 12: np.array([245, 130, 48]),
                 13: np.array([245, 130, 48]),
                 14: np.array([128, 128, 0]),
                 15: np.array([128, 128, 0]),
                 16: np.array([210, 245, 60]),
                 17: np.array([210, 245, 60]),
                 18: np.array([170, 255, 195]),
                 19: np.array([60, 180, 75]),
                 20: np.array([60, 180, 75]),
                 21: np.array([0, 128, 128]),
                 22: np.array([0, 130, 200]),
                 23: np.array([0, 130, 200]),
                 24: np.array([240, 50, 230]),
                 25: np.array([240, 50, 230])}