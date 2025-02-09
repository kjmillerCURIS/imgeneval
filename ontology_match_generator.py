import os
import sys
from itertools import product, permutations
from collections import Counter


def valid_partition(partition, K):
    """
    Checks if a given partition satisfies the condition that if any set has >2 elements,
    then no set can have <2 elements.
    """
    # Count occurrences of each set
    counts = Counter(partition)
    
    # If any set has more than 2 items, then all must have at least 2, otherwise we underserved a group that we could've served
    if any([counts[k] > 2 for k in range(K)]):
        return all([counts[k] >= 2 for k in range(K)])
    
    return True  # Otherwise, it's valid


#N is number of detections, K is number of group objects
def generate_partitions(N, K):
    """
    Generates all valid partitions of N items into K sets.
    
    A partition is represented as a list of length N, where the ith element
    is an integer in {0, ..., K-1} representing the set assignment of item i.
    """
    all_partitions = product(range(K), repeat=N)  # Generate all possible assignments
    
    # Filter partitions based on the validity rule
    valid_partitions = [p for p in all_partitions if valid_partition(p, K)]
    
    return valid_partitions


#N is number of detections, K is number of single objects
#if K > N then we swap N and K and then transpose the answer somehow
def generate_matches(N, K):
    if K > N:
        trans_matches = generate_matches(K, N)
        matches = []
        for tm in trans_matches:
            match = [-1] * N
            for k, n in enumerate(tm):
                if n >= 0:
                    assert(match[n] == -1)
                    match[n] = k

            matches.append(match)

        return matches

    my_perms = permutations(range(N), r=K)
    matches = []
    for p in my_perms:
        match = [-1] * N
        for k, pp in enumerate(p):
            match[pp] = k

        matches.append(match)

    return matches


#objects should be list of objects, all single
#detections should be list of detections
def match_generator_one_name_single(objects, detections):
    assert(len(objects) > 0)
    if len(detections) == 0:
        yield {my_object['object_id'] : None for my_object in objects}
        return

    matches = generate_matches(len(detections), len(objects))
    for match in matches:
        full_match = {}
        for n, k in enumerate(match):
            if k >= 0:
                assert(objects[k]['object_id'] not in full_match)
                full_match[objects[k]['object_id']] = detections[n]

        for my_object in objects:
            if my_object['object_id'] not in full_match:
                full_match[my_object['object_id']] = None

        yield full_match


#objects should be list of objects, all group
#detections should be list of detections
def match_generator_one_name_group(objects, detections):
    assert(len(objects) > 0)
    if len(detections) == 0:
        yield {my_object['object_id'] : [] for my_object in objects}
        return

    partitions = generate_partitions(len(detections), len(objects))
    for partition in partitions:
        full_match = {}
        for n, k in enumerate(partition):
            if k >= 0:
                if objects[k]['object_id'] not in full_match:
                    full_match[objects[k]['object_id']] = []

                full_match[objects[k]['object_id']].append(detections[n])

        for my_object in objects:
            if my_object['object_id'] not in full_match:
                full_match[my_object['object_id']] = None

        yield full_match


#expect objects to be dict mapping each object_id to its object
#expect detections to be dict mapping each object name to a list of detections
#yield a dictionary mapping from object_id to either detection or list of detections or None
def match_generator(objects, detections):
    names2objects_single = {}
    names2objects_group = {}
    for k in sorted(objects.keys()):
        name = objects[k]['name']
        if objects[k]['is_group']:
            if name not in names2objects_group:
                names2objects_group[name] = []
            names2objects_group[name].append(objects[k])
        else:
            if name not in names2objects_single:
                names2objects_single[name] = []
            names2objects_single[name].append(objects[k])

    genny_list = []
    for name in sorted(names2objects_single.keys()):
        assert(len(names2objects_single[name]) > 0)
        genny_list.append(match_generator_one_name_single(names2objects_single[name], detections[name]))

    for name in sorted(names2objects_group.keys()):
        assert(len(names2objects_group[name]) > 0)
        genny_list.append(match_generator_one_name_group(names2objects_group[name], detections[name]))

    for t in product(*genny_list):
        match = {}
        for tt in t:
            for k in sorted(tt.keys()):
                assert(k not in match)
                match[k] = tt[k]

        yield match
