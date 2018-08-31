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


def investigate_attention(attn, attn_pos, sentence_words, outer_vocab):

    ########################################
    # options for the investigations BEGIN #
    ########################################

    # choose which head to investigate
    # head = 3.0
    list_of_head = [1.0, 2.0, 3.0]
    list_of_combinations = ["attn", "attn_pos", "combined"]
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
                        unpadded_attention = numpy_attention[sentence_index + 50][:first_pad_position]
                    elif head == 3.0:
                        unpadded_attention = numpy_attention[sentence_index + 100][:first_pad_position]
                else:
                    unmapped_sentence_final = unmapped_sentence
                    # attention heads, 1: +0 2: +50 3: +100. i.e. head 2: numpy_attention[sentence_index+50]...
                    if head == 1.0:
                        unpadded_attention = numpy_attention[sentence_index]  # numpy_attention[sentence_index+50]
                    elif head == 2.0:
                        unpadded_attention = numpy_attention[sentence_index + 50]  # numpy_attention[sentence_index+50]
                    elif head == 3.0:
                        # print(numpy_attention)
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
