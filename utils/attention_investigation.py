import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
import operator

from global_random_seed import RANDOM_SEED
# make everything reproducible
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# Already implemented: Ausgabe mean position/variance der scores
# 1. rausfinden, welche axis wörter, welche axis sind die positionen
# 2. für jedes über die positionen axis softmax (wird nur für zusätzliche Analyse verwendet)
# 3. Pro Wort: w = softmax(attention_scores), r = alle positions = "np.arange(len(w))"
# 4. mean =  weighted_average = np.average(r, weights=w)
# 5. std_dev = https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
# 6. Verteilung pro Satz ausgeben, für folgende Wörter (und erste 20 Sätze):
#   - größtes/kleinstes mean
#   - größte/kleinste std_dev


# TODO: finish me
def plot_generator_with_softmax(data):

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid")

    fig = plt.figure(figsize=(8, 6))

    # Not showing the data lists here

    def softmax(x):
        return np.exp(x) / np.exp(x).sum(axis=0)

    # attn
    data = [-0.005951299, -0.029044457, -2.4142258, 2.190176, 1.6590111, -3.6578588, -4.684568, -3.6578588, -3.6578588,
            6.084622, 13.555143, 13.555143, -4.34307, -1.1962872, -0.47124174, -0.43637118, -8.555032, -6.269103,
            -4.5881658, -5.9282355, -4.585698, -5.2294097, -6.944389, -3.5933948, -4.3854713, -2.5584598, -4.093828,
            -7.9165735, -4.7697864, -5.935749, -5.9056034, -3.213047, -8.166379, -5.539582, -6.075837, -4.260547,
            -2.8777754, -3.6983604, -4.148365, -4.4483614, -2.085313, -3.1456637, -3.706437, 2.0111775, 2.0111775,
            2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775,
            2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775,
            2.0111775, 2.0111775]

    # attn_pos
    data2 = [0.77760446, -0.3654992, -0.3654992, -0.3654992, -0.3654992, -0.3654992, -0.3654992, -0.3654992, -0.3654992,
             0.5600485, 0.5600485, 0.5600485, 0.5600485, 1.0015192, 1.0015192, -0.8097563, -0.14138925, 0.66161174,
             -0.43525112, -0.43525112, 0.29473847, 0.29473847, 0.30402362, 0.4407499, 0.99283254, 1.0506727, 1.0493455,
             1.0368179, 1.0545973, 1.1171892, 1.0651786, 1.1270455, 0.8863518, 0.58156127, 0.6494905, 0.7443865,
             0.69479305, 0.85680157, 0.63184834, 0.45794958, 0.38933513, 0.1833995, 0.32618314, 0.31002286, 0.23198071,
             0.53673124, 0.34084955, 0.2578649, -0.8240579, -1.3246806, -1.287403, -1.3355032, -1.1736788, -1.1917881,
             -1.1917881, -1.7546477, -1.5299661, -1.5299661, -1.6786951, -1.5299661, -1.1985171, -0.49476224,
             -0.4208252, -0.1696207, -0.33454028]

    # combined
    data3 = [0.7716532, -0.39454365, -2.779725, 1.8246768, 1.2935119, -4.023358, -5.050067, -4.023358, -4.023358,
             6.6446705, 14.115191, 14.115191, -3.7830215, -0.19476795, 0.5302775, -1.2461275, -8.696421, -5.6074915,
             -5.023417, -6.363487, -4.29096, -4.9346714, -6.6496506, -3.2986562, -4.2205296, -2.393518, -3.9288864,
             -7.7516317, -4.6048446, -5.7708073, -5.7406616, -3.0481052, -8.452318, -5.8255215, -6.361777, -4.546487,
             -3.163715, -3.9843, -4.4343047, -4.734301, -2.3712525, -3.4316032, -3.9923766, 1.725238, 1.725238,
             1.725238, 1.725238, 1.725238, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384,
             0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384,
             0.74440384, 0.74440384]

    softmax_flag = True

    if softmax_flag:
        # plt.bar(range(len(data)), softmax(data), color='b', alpha=0.3, label ='attn') #, hatch="/")
        # plt.bar(range(len(data2)), softmax(data2), color='g', alpha = 0.3, label ='attn_rel_pos') #, hatch="o")
        # plt.bar(range(len(data3)), softmax(data3), color='r',alpha = 0.3, label ='attn_comb') #, hatch="\\")
        plt.plot(range(len(data[:43])), softmax(data[:43]), '-bx', alpha=0.5, label=r'$softmax(y_{inner})$')
        plt.plot(range(len(data2[:43])), softmax(data2[:43]), '-go', alpha=0.5, lw=2, ms=4,
                 label=r'$softmax(y_{rel\_pos})$')
        plt.plot(range(len(data3[:43])), softmax(data3[:43]), '-r+', alpha=0.5, lw=2,
                 label=r'$softmax(y_{inner} + y_{rel\_pos})$')
        ax = fig.add_subplot(111)
        ax.fill_between(range(len(data[:43])), 0, softmax(data[:43]), color='dodgerblue', alpha=0.4)
        ax.fill_between(range(len(data[:43])), 0, softmax(data2[:43]), color='mediumseagreen', alpha=0.4)
        ax.fill_between(range(len(data[:43])), 0, softmax(data3[:43]), color='indianred', alpha=0.4)

    plt.title("Head 1")
    plt.xlabel("Word Position in the Sentence")
    plt.ylabel("Attention Probability")

    # plt.grid()
    plt.legend(fontsize=16)

    plt.savefig('head_1_in_32_softmax_lines.png', dpi=350)

    # plt.close(fig)


# TODO: finish me
def plot_generator_without_softmax(data):

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid")

    fig = plt.figure(figsize=(8, 6))

    # Not showing the data lists here

    def softmax(x):
        return np.exp(x) / np.exp(x).sum(axis=0)

    # attn
    data = [-0.005951299, -0.029044457, -2.4142258, 2.190176, 1.6590111, -3.6578588, -4.684568, -3.6578588, -3.6578588,
            6.084622, 13.555143, 13.555143, -4.34307, -1.1962872, -0.47124174, -0.43637118, -8.555032, -6.269103,
            -4.5881658, -5.9282355, -4.585698, -5.2294097, -6.944389, -3.5933948, -4.3854713, -2.5584598, -4.093828,
            -7.9165735, -4.7697864, -5.935749, -5.9056034, -3.213047, -8.166379, -5.539582, -6.075837, -4.260547,
            -2.8777754, -3.6983604, -4.148365, -4.4483614, -2.085313, -3.1456637, -3.706437, 2.0111775, 2.0111775,
            2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775,
            2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775, 2.0111775,
            2.0111775, 2.0111775]

    # attn_pos
    data2 = [0.77760446, -0.3654992, -0.3654992, -0.3654992, -0.3654992, -0.3654992, -0.3654992, -0.3654992, -0.3654992,
             0.5600485, 0.5600485, 0.5600485, 0.5600485, 1.0015192, 1.0015192, -0.8097563, -0.14138925, 0.66161174,
             -0.43525112, -0.43525112, 0.29473847, 0.29473847, 0.30402362, 0.4407499, 0.99283254, 1.0506727, 1.0493455,
             1.0368179, 1.0545973, 1.1171892, 1.0651786, 1.1270455, 0.8863518, 0.58156127, 0.6494905, 0.7443865,
             0.69479305, 0.85680157, 0.63184834, 0.45794958, 0.38933513, 0.1833995, 0.32618314, 0.31002286, 0.23198071,
             0.53673124, 0.34084955, 0.2578649, -0.8240579, -1.3246806, -1.287403, -1.3355032, -1.1736788, -1.1917881,
             -1.1917881, -1.7546477, -1.5299661, -1.5299661, -1.6786951, -1.5299661, -1.1985171, -0.49476224,
             -0.4208252, -0.1696207, -0.33454028]

    # combined
    data3 = [0.7716532, -0.39454365, -2.779725, 1.8246768, 1.2935119, -4.023358, -5.050067, -4.023358, -4.023358,
             6.6446705, 14.115191, 14.115191, -3.7830215, -0.19476795, 0.5302775, -1.2461275, -8.696421, -5.6074915,
             -5.023417, -6.363487, -4.29096, -4.9346714, -6.6496506, -3.2986562, -4.2205296, -2.393518, -3.9288864,
             -7.7516317, -4.6048446, -5.7708073, -5.7406616, -3.0481052, -8.452318, -5.8255215, -6.361777, -4.546487,
             -3.163715, -3.9843, -4.4343047, -4.734301, -2.3712525, -3.4316032, -3.9923766, 1.725238, 1.725238,
             1.725238, 1.725238, 1.725238, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384,
             0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384, 0.74440384,
             0.74440384, 0.74440384]

    softmax_flag = False

    if softmax_flag:
        plt.bar(range(len(data)), softmax(data), color='b', alpha=0.3, label='attn')  # , hatch="/")
        plt.bar(range(len(data2)), softmax(data2), color='g', alpha=0.3, label='attn_rel_pos')  # , hatch="o")
        plt.bar(range(len(data3)), softmax(data3), color='r', alpha=0.3, label='attn_comb')  # , hatch="\\")
        plt.plot(range(len(data)), softmax(data), 'bx', alpha=0.5)
        plt.plot(range(len(data2)), softmax(data2), 'go', alpha=0.5, ms=4)
        plt.plot(range(len(data3)), softmax(data3), 'r+', alpha=0.5)
    else:
        plt.bar(range(len(data[:43])), data[:43], color='b', alpha=0.3, label=r'$y_{inner}$')  # , hatch="/")
        plt.bar(range(len(data2[:43])), data2[:43], color='g', alpha=0.3, label=r'$y_{rel\_pos}$')  # , hatch="o")
        plt.bar(range(len(data3[:43])), data3[:43], color='r', alpha=0.3,
                label=r'$y_{inner} + y_{rel\_pos}$')  # , hatch="\\")
        plt.plot(range(len(data[:43])), data[:43], 'bx', alpha=0.5)
        plt.plot(range(len(data2[:43])), data2[:43], 'go', alpha=0.5, ms=4)
        plt.plot(range(len(data3[:43])), data3[:43], 'r+', alpha=0.5)

    plt.title("Head 1")
    plt.xlabel("Word Position in the Sentence")
    plt.ylabel("Attention Weight")

    # plt.grid()
    plt.legend(fontsize=16)

    plt.savefig('head_1_in_32_no_softmax.png', dpi=350)

    # plt.close(fig)


def investigate_attention(attn, attn_pos, sentence_words, outer_vocab):

    ########################################
    # options for the investigations BEGIN #
    ########################################

    # choose which head to investigate
    # head = 3.0
    list_of_head = [1.0, 2.0, 3.0]
    list_of_combinations = ["attn", "attn_pos", "combined"]

    # save data for plots
    # TODO: generate plots automatically
    of_data = dict()
    in_32_data = dict()
    in_16_data = dict()

    for head in list_of_head:
        for what_attention_to_investigate in list_of_combinations:
            # options to investigate from:
            #   attn:  basic attention from self-attention paper without any pos encodings
            #   attn_pos:  attention with our relative pos encodings
            #   combined:  both of the above combined
            # what_attention_to_investigate = "combined"

            if what_attention_to_investigate == "combined":
                all_attn = attn + attn_pos.transpose(1, 2)

            numpy_sentences = sentence_words.cpu().numpy()

            if what_attention_to_investigate == "attn":
                numpy_attention = attn.cpu().numpy()
            elif what_attention_to_investigate == "attn_pos":
                numpy_attention = attn_pos.cpu().numpy()
            elif what_attention_to_investigate == "combined":
                numpy_attention = all_attn.cpu().numpy()

            # generate file to save output to based on params above
            investigations_filename = "".join(
                [
                    'saved_models/out/attention_mean_and_std_head_',
                    str(int(head)),
                    '_',
                    str(what_attention_to_investigate),
                    '.txt'
                ]
            )

            ######################################
            # options for the investigations END #
            ######################################

            def softmax(x):
                """Compute softmax values for each sets of scores in x."""
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum(axis=0)  # only difference

            def weighted_avg_and_std(values, weights):
                """
                Return the weighted average and standard deviation.
                values, weights -- Numpy ndarrays with the same shape.
                """

                # have to substract the index position from the mean
                # TODO: in mean and std

                average = np.average(values, weights=weights)
                # Fast and numerically precise:
                variance = np.average((values - average) ** 2, weights=weights)
                return average, math.sqrt(variance)

            # following run is used for the investigations: python eval.py --model_dir saved_models/lm4
            # --model checkpoint_epoch_60.pt
            sentence_to_search = "They cited the case of OBJ-ORGANIZATION OBJ-ORGANIZATION OBJ-ORGANIZATION OBJ-ORGANIZATION subcontractor SUBJ-PERSON SUBJ-PERSON , who was working in Cuba on a tourist visa and possessed satellite communications equipment , who has been held in a maximum security prison since his arrest Dec 3 ."
            for sentence_index, each_sentence in enumerate(numpy_sentences):

                unmapped_sentence = outer_vocab.unmap(each_sentence)

                # DONe
                # TODO: skip the padded words in both vectors

                # get id-to-word sentence representation

                # print(unmapped_sentence, len(unmapped_sentence))

                # find the position of the first pad
                first_pad_position = None
                for i, single_word in enumerate(unmapped_sentence):
                    if single_word == '<PAD>':
                        first_pad_position = i
                        break

                if first_pad_position:
                    # get all words before the first pad appears
                    unmapped_sentence_final = unmapped_sentence[:first_pad_position]  # [:first_pad_position]
                    # iterate over attention scores
                    # attention heads, 1: +0 2: +50 3: +100. i.e. head 2: numpy_attention[sentence_index+50]...
                    if head == 1.0:
                        unpadded_attention = numpy_attention[sentence_index][:first_pad_position]
                    elif head == 2.0:
                        if len(numpy_attention) != 27:  # skip the last batch of size 27
                            unpadded_attention = numpy_attention[sentence_index + 50][:first_pad_position]
                    elif head == 3.0:
                        if len(numpy_attention) != 27:  # skip the last batch of size 27
                            unpadded_attention = numpy_attention[sentence_index + 100][:first_pad_position]
                else:
                    unmapped_sentence_final = unmapped_sentence
                    # attention heads, 1: +0 2: +50 3: +100. i.e. head 2: numpy_attention[sentence_index+50]...
                    if head == 1.0:
                        unpadded_attention = numpy_attention[sentence_index]  # numpy_attention[sentence_index+50]
                    elif head == 2.0:
                        if len(numpy_attention) != 27:  # skip the last batch of size 27
                            unpadded_attention = numpy_attention[sentence_index + 50]  # numpy_attention[sentence_index+50]
                    elif head == 3.0:
                        # print(numpy_attention)
                        if len(numpy_attention) != 27:  # skip the last batch of size 27
                            unpadded_attention = numpy_attention[sentence_index + 100]  # numpy_attention[sentence_index+50]

                # TODO: get rid of this later
                # print(" ".join(unmapped_sentence))

                if " ".join(unmapped_sentence_final) == sentence_to_search:

                    # make sure the slicing was correct on both tensors
                    assert len(unmapped_sentence_final) == len(unpadded_attention)

                    # print(len(unpadded_attention), len(unmapped_sentence_final))
                    print(unmapped_sentence_final, len(unmapped_sentence_final))
                    # print(unpadded_attention, len(unpadded_attention))
                    print()

                    all_means = []
                    all_stds = []

                    highest_attention_score_per_word = dict()

                    for i, each_word in enumerate(unmapped_sentence_final):
                        # print("WORD:", each_word)
                        # print("POS. VECTOR:", unpadded_attention[i])

                        # for each word pos vector, select the highest attention score to other word but itself

                        # mask out the word itself

                        if first_pad_position:
                            current_word_pos_vector = unpadded_attention[i][:first_pad_position]
                        else:
                            current_word_pos_vector = unpadded_attention[i]

                        indices = [i]
                        mask = np.zeros(current_word_pos_vector.size, dtype=bool)
                        mask[indices] = True
                        a = np.ma.array(current_word_pos_vector, mask=mask)

                        # select highest attention score for given word
                        # print(len(a))
                        highest_attention_index, highest_attention_score = max(enumerate(a), key=operator.itemgetter(1))
                        highest_attention_score_per_word[i] = (highest_attention_index, highest_attention_score)
                        # print(i, highest_attention_index, highest_attention_score)

                        # get weighted average
                        w = softmax(current_word_pos_vector)
                        # print(w, len(w))
                        r = np.arange(len(w)) - i

                        weighted_average_and_std = weighted_avg_and_std(r, w)

                        # print("WORD:", each_word)
                        # print("Weighted Average:", weighted_average_and_std[0])
                        # print("Weighted Standard Deviation:", weighted_average_and_std[1])

                        # print(i, first_pad_position, unmapped_sentence_final)

                        if first_pad_position:
                            if i < first_pad_position:
                                all_means.append(weighted_average_and_std[0])
                                all_stds.append(weighted_average_and_std[1])
                        else:
                            all_means.append(weighted_average_and_std[0])
                            all_stds.append(weighted_average_and_std[1])

                        # highest_attention_score_per_word[i] = max(enumerate(all_means), key=operator.itemgetter(1))
                    # print(all_means)

                    smallest_mean_index, smallest_mean = min(enumerate(all_means), key=operator.itemgetter(1))
                    biggest_mean_index, biggest_mean = max(enumerate(all_means), key=operator.itemgetter(1))

                    print()
                    print("without padding")
                    print("smallest_mean:", smallest_mean)
                    print("smallest_mean_word:", unmapped_sentence_final[smallest_mean_index], "index:", smallest_mean_index)
                    print("^ av_mean:", all_means[smallest_mean_index], "av_std:", all_stds[smallest_mean_index])
                    print(
                        "highest attention score:", highest_attention_score_per_word[smallest_mean_index][1], "word:",
                        unmapped_sentence_final[highest_attention_score_per_word[smallest_mean_index][0]],
                        "index:", highest_attention_score_per_word[smallest_mean_index][0])
                    print()

                    print("biggest_mean:", biggest_mean)
                    print("biggest_mean_word:", unmapped_sentence_final[biggest_mean_index], "index:", biggest_mean_index)
                    print("^ av_mean:", all_means[biggest_mean_index], "av_std:", all_stds[biggest_mean_index])
                    print(
                        "highest attention score:", highest_attention_score_per_word[biggest_mean_index][1], "word:",
                        unmapped_sentence_final[highest_attention_score_per_word[biggest_mean_index][0]],
                        "index:", highest_attention_score_per_word[biggest_mean_index][0])
                    print()

                    smallest_std_index, smallest_std = min(enumerate(all_stds), key=operator.itemgetter(1))
                    biggest_std_index, biggest_std = max(enumerate(all_stds), key=operator.itemgetter(1))

                    print("smallest_std:", smallest_std)
                    print("smallest_std_word:", unmapped_sentence_final[smallest_std_index], "index:", smallest_std_index)
                    print("^ av_mean:", all_means[smallest_std_index], "av_std:", all_stds[smallest_std_index])
                    print(
                        "highest attention score:", highest_attention_score_per_word[smallest_std_index][1], "word:",
                        unmapped_sentence_final[highest_attention_score_per_word[smallest_std_index][0]],
                        "index:", highest_attention_score_per_word[smallest_std_index][0])
                    print()

                    print("biggest_std:", biggest_std)
                    print("biggest_std_word:", unmapped_sentence_final[biggest_std_index], "index:", biggest_std_index)
                    print("^ av_mean:", all_means[biggest_std_index], "av_std:", all_stds[biggest_std_index])
                    print(
                        "highest attention score:", highest_attention_score_per_word[biggest_std_index][1], "word:",
                        unmapped_sentence_final[highest_attention_score_per_word[biggest_std_index][0]],
                        "index:", highest_attention_score_per_word[biggest_std_index][0])
                    print()

                    with open(investigations_filename, 'a') as outfile:

                        # print each att score
                        outfile.write(" ".join(
                            [" ".join(unmapped_sentence_final), "\n", str(len(unmapped_sentence_final)), "\n"]))

                        for i, element in enumerate(unmapped_sentence_final):
                            outfile.write(" ".join(["index:", str(i), " /// word:", str(element), "\n"]))

                            outfile.write(" ".join(
                                ["Attention vector:", "[", ", ".join([str(x) for x in unpadded_attention[i]]), "]", "\n",
                                 str(len(unpadded_attention)), "\n",
                                 "av_mean: ", str(all_means[i]), "av_str: ", str(all_stds[i]), "\n"]))

                            outfile.write(
                                " ".join(["[", ", ".join([str(x) for x in softmax(current_word_pos_vector)]), "]", "\n"]))

                        # outfile.write(" ".join([" ".join([str(x) for x in unpadded_attention]), "\n", str(len(unpadded_attention)), "\n"]))

                        outfile.write(
                            " ".join([" ".join(unmapped_sentence_final), "\n", str(len(unmapped_sentence_final)), "\n"]))
                        outfile.write(" ".join(["without padding", "\n"]))
                        outfile.write(" ".join(["smallest_mean:", str(smallest_mean), "\n"]))
                        outfile.write(" ".join(
                            ["smallest_mean_word:", str(unmapped_sentence_final[smallest_mean_index]), " index:",
                             str(smallest_mean_index), "\n"]))
                        outfile.write(" ".join(
                            ["^ av_mean:", str(all_means[smallest_mean_index]), "av_std:", str(all_stds[smallest_mean_index]),
                             "\n"]))
                        outfile.write(" ".join([
                            "highest attention score:", str(highest_attention_score_per_word[smallest_mean_index][1]), "word:",
                            str(unmapped_sentence_final[highest_attention_score_per_word[smallest_mean_index][0]]),
                            "index:", str(highest_attention_score_per_word[smallest_mean_index][0]), "\n", "\n"]))

                        outfile.write(" ".join(["biggest_mean:", str(biggest_mean), "\n"]))
                        outfile.write(" ".join(
                            ["biggest_mean_word:", str(unmapped_sentence_final[biggest_mean_index]), " index:",
                             str(biggest_mean_index), "\n"]))
                        outfile.write(" ".join(["^ av_mean:", str(all_means[biggest_mean_index]), "av_std:",
                                                str(all_stds[biggest_mean_index]), "\n"]))
                        outfile.write(" ".join([
                            "highest attention score:", str(highest_attention_score_per_word[biggest_mean_index][1]),
                            "word:",
                            str(unmapped_sentence_final[highest_attention_score_per_word[biggest_mean_index][0]]),
                            "index:", str(highest_attention_score_per_word[biggest_mean_index][0]), "\n", "\n"]))

                        outfile.write(" ".join(["smallest_std:", str(smallest_std), "\n"]))
                        outfile.write(" ".join(
                            ["smallest_std_word:", str(unmapped_sentence_final[smallest_std_index]), " index:",
                             str(smallest_std_index), "\n"]))
                        outfile.write(" ".join(["^ av_mean:", str(all_means[smallest_std_index]), "av_std:",
                                                str(all_stds[smallest_std_index]), "\n"]))
                        outfile.write(" ".join([
                            "highest attention score:", str(highest_attention_score_per_word[smallest_std_index][1]),
                            "word:",
                            str(unmapped_sentence_final[highest_attention_score_per_word[smallest_std_index][0]]),
                            "index:", str(highest_attention_score_per_word[smallest_std_index][0]), "\n", "\n"]))

                        outfile.write(" ".join(["biggest_std:", str(biggest_std), "\n"]))
                        outfile.write(" ".join(["biggest_std_word:", str(unmapped_sentence_final[biggest_std_index]), " index:",
                                                str(biggest_std_index), "\n"]))
                        outfile.write(" ".join(["^ av_mean:", str(all_means[biggest_std_index]), "av_std:",
                                                str(all_stds[biggest_std_index]), "\n"]))
                        outfile.write(" ".join([
                            "highest attention score:", str(highest_attention_score_per_word[biggest_std_index][1]),
                            "word:",
                            str(unmapped_sentence_final[highest_attention_score_per_word[biggest_std_index][0]]),
                            "index:", str(highest_attention_score_per_word[biggest_std_index][0]), "\n", "\n"]))

                        outfile.write("\n")
