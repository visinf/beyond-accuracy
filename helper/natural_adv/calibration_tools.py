import numpy as np
import sklearn.metrics as sk
recall_level_default = 0.95

def calib_err(confidence, correct, p='2', beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr


def aurra(confidence, correct):
    conf_ranks = np.argsort(confidence)[::-1]  # indices from greatest to least confidence
    rra_curve = np.cumsum(np.asarray(correct)[conf_ranks])
    rra_curve = rra_curve / np.arange(1, len(rra_curve) + 1)  # accuracy at each response rate
    return np.mean(rra_curve)


def soft_f1(confidence, correct):
    wrong = 1 - correct

    # # the incorrectly classified samples are our interest
    # # so they make the positive class
    # tp_soft = np.sum((1 - confidence) * wrong)
    # fp_soft = np.sum((1 - confidence) * correct)
    # fn_soft = np.sum(confidence * wrong)

    # return 2 * tp_soft / (2 * tp_soft + fn_soft + fp_soft)
    return 2 * ((1 - confidence) * wrong).sum()/(1 - confidence + wrong).sum()

def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    aupr = sk.average_precision_score(labels, examples)

    return aupr


def print_measures_old(auroc, aupr, fpr, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t' + method_name)
    print('FPR{:d}:\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t{:.2f}'.format(100 * aupr))
