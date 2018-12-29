from data.loader import DataLoader
from utils.vocab import Vocab

vocab_file = '../dataset/vocab/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
data_loader = DataLoader("../dataset/tacred/train.json",
                         50,
                         {"preload_lemmas": False, "use_lemmas": False, "lower": True, "relative_positions": True},
                         vocab
                         )

# test 1
positional_vector = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
expected_result = [-2, -2, -1, 0, 1, 2, 2, 3, 3, 3, 3, 4]

bin_positions_result = data_loader.bin_positions(positional_vector)
assert expected_result == bin_positions_result

# test 2
positional_vector = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
expected_result = [-4, -4, -4, -3, -3, -3, -3, -2, -2, -1, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4]

bin_positions_result = data_loader.bin_positions(positional_vector)
assert expected_result == bin_positions_result

# test 3
positional_vector = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
expected_result = [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]

bin_positions_result = data_loader.bin_positions(positional_vector)
assert expected_result == bin_positions_result

# test 4
positional_vector = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]
expected_result = [-4, -4, -4, -3, -3, -3, -3, -2, -2, -1, 0]

bin_positions_result = data_loader.bin_positions(positional_vector)
assert expected_result == bin_positions_result