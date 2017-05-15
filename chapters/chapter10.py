def bonferroni(ps, alpha):
    """
    Return a p-value for multiple tests using Bonferroni.

    See page 166.
    """
    return alpha / len(ps)


def benjamini_hochberg(ps, alpha, independent=True):
    """
    Return a p-value for multiple tests using the Benjamini-Hochberg method.

    See page 167.
    """

    m = len(ps)
    ps = sorted(ps)

    if independent:
        c = 1
    else:
        c = sum(1 / i for i in range(1, m + 1))

    ls = [i * alpha / (c * m) for i in range(1, m + 1)]
    r = max(i for i in range(m) if ps[i] < ls[i])

    return ps[r]
