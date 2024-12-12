def offset_indexer(n):
    """Calculates the global offsets indices.

    e.g. n=3:
        (i, i+1), (j, j+1)
        [[(0, 1), (0, 1)], [(0, 1), (1, 2)], [(0, 1), (2, 3)],
         [(1, 2), (1, 2)], [(1, 2), (2, 3)], [(2, 3), (2, 3)]]
         - the row indices for the first comparison are (0,1)
         - the column indices for the first comparison are (0,1)
         - etc...
    """
    offset_indices = []
    for i in range(n):
        for j in range(n):
            if j >= i:
                offset_index = [(i, i + 1), (j, j + 1)]
                offset_indices.append(offset_index)
    return offset_indices
