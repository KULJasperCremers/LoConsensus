from typing import Generator

import numpy as np
from locomotif.loconsensus import candidate_finder as cf
from locomotif.loconsensus import path as path_class
from locomotif.loconsensus import path_finder as pf

OVERLAP = 0.0


def find_motifs_representativesV1(
    max_amount: int,
    n: int,
    m: int,
    paths: list[path_class.Path],
    mirrored_paths: list[path_class.Path],
    L_MIN: int,
    L_MAX: int,
) -> Generator[
    tuple[tuple[int, int], list[path_class.Path], list[tuple[int, int]], str, float],
    None,
    None,
]:
    """Generate motifs by finding and masking the best candidates based on fitness scores."""
    # requires masks for each length to find the motifs in all directions
    column_start_mask = np.full(n, True)
    column_end_mask = np.full(n, True)
    column_mask = np.full(n, False)
    row_start_mask = np.full(m, True)
    row_end_mask = np.full(m, True)
    row_mask = np.full(m, False)
    amount = 0

    while max_amount is None or amount < max_amount:
        # break if all positions are masked or no valid start/end positions are left
        if (
            np.all(column_mask)
            or not np.any(column_start_mask)
            or not np.any(column_end_mask)
        ) and (
            np.all(row_mask) or not np.any(row_start_mask) or not np.any(row_end_mask)
        ):
            break

        # update masks to exclude already masked positions
        column_start_mask &= ~column_mask
        column_end_mask &= ~column_mask
        row_start_mask &= ~row_mask
        row_end_mask &= ~row_mask

        # check the paths for candidates in column POV
        column_best_candidate, column_best_fitness = cf.find_candidatesV1(
            column_start_mask,
            column_end_mask,
            column_mask,
            paths,
            L_MIN,
            L_MAX,
            OVERLAP,
        )

        # check the mirrored paths for candidates in row POV
        row_best_candidate, row_best_fitness = cf.find_candidatesV1(
            row_start_mask,
            row_end_mask,
            row_mask,
            mirrored_paths,
            L_MIN,
            L_MAX,
            OVERLAP,
        )

        if column_best_fitness >= row_best_fitness and column_best_fitness > 0.0:
            best_candidate = column_best_candidate
            current_paths = paths
            current_mask = column_mask
            current_length = n
            best_fitness = column_best_fitness
            pov = 'column'
        elif row_best_fitness >= column_best_fitness and row_best_fitness > 0.0:
            best_candidate = row_best_candidate
            current_paths = mirrored_paths
            current_mask = row_mask
            current_length = m
            best_fitness = row_best_fitness
            pov = 'row'
        else:
            break

        (start_index, end_index) = best_candidate
        induced_paths = pf.find_induced_paths(
            start_index, end_index, current_paths, current_mask
        )
        motif_set = [(path[0][0], path[-1][0] + 1) for path in induced_paths]

        for motif_start, motif_end in motif_set:
            motif_length = motif_end - motif_start
            overlap = int(OVERLAP * motif_length)

            # ensure valid adjusted indices that account for overlap
            start_index = motif_start + overlap
            end_index = motif_end - overlap
            start_index = max(0, start_index)
            end_index = min(current_length, end_index)

            # skip if adjusted indices are invalid
            if start_index >= end_index:
                continue

            current_mask[start_index:end_index] = True

        amount += 1
        yield best_candidate, induced_paths, motif_set, pov, best_fitness
