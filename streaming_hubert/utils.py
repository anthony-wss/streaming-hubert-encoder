import numpy as np

def cal_duplicate_tokens(ssl_units):
    """
        Return the mean and std of the average duplicate token lengths.

        Parameters:
            ssl_units (list(list(int))): lists of discrete token sequences.
    """
    results = []
    for seq in ssl_units:
        dup_token_lens = []
        cur_len = 1
        for i in range(len(seq)):
            if i+1 < len(seq):
                if seq[i] == seq[i+1]:
                    cur_len += 1
                else:
                    dup_token_lens.append(cur_len)
                    cur_len = 1
            else:
                dup_token_lens.append(cur_len)
        results.append(np.mean(dup_token_lens))
    return np.mean(results), np.std(results)

