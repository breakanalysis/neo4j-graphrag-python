import random
import string
import uuid
import time
from scipy.cluster.hierarchy import DisjointSet

DEGREE = 25
REL_TYPES_PER_UNIT = 500

def random_description(length=8):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def merge_descriptions(desc1, desc2):
    return random.choice([desc1, desc2])

def main():
    total_elements = 1_000_000
    subset_size = 1000
    num_subsets = total_elements // subset_size
    merge_prob = 0.99
    print(f"Ingesting {total_elements} nodes divided into subsets of {subset_size} nodes.")
    print(f"In each subset, the average degree is {DEGREE} and relationship is sampled from {REL_TYPES_PER_UNIT} random types")
    print(f"Both nodes and relationship types are merged with a previously seen such object with probability {merge_prob}")

    # Node disjoint set + descriptions
    ds = DisjointSet()
    node_descriptions = {}

    # Relationship type disjoint set + descriptions
    rt_ds = DisjointSet()
    rel_descriptions = {}

    # Step 1: generate input sets, rel types, and triples
    input_sets = []
    subset_rel_types = []
    triples = []

    for _ in range(num_subsets):
        nodes = [uuid.uuid4() for _ in range(subset_size)]
        rels = [uuid.uuid4() for _ in range(REL_TYPES_PER_UNIT)]

        for node in nodes:
            node_descriptions[node] = random_description()
        for rel in rels:
            rel_descriptions[rel] = random_description()

        for src in nodes:
            for _ in range(DEGREE):
                trg = random.choice(nodes)
                rel = random.choice(rels)
                triples.append((src, rel, trg))

        input_sets.append(nodes)
        subset_rel_types.append(rels)

    print(f"Generated {len(triples):,} triples.")

    # Step 2: build DS and do merges
    elem_num = 0
    rel_num = 0
    t0 = time.time()

    for set_id, input_set in enumerate(input_sets):
        # process nodes
        for elem in input_set:
            ds.add(elem)
            elem_num += 1

            if elem_num > 1 and random.random() < merge_prob:
                other = random.randint(0, elem_num - 1)
                other_elem = input_sets[other // subset_size][other % subset_size]

                rep1 = ds[elem]
                rep2 = ds[other_elem]

                if rep1 != rep2:
                    ds.merge(rep1, rep2)
                    new_rep = ds[rep1]

                    merged_desc = merge_descriptions(
                        node_descriptions[rep1], node_descriptions[rep2]
                    )
                    node_descriptions[new_rep] = merged_desc
                    if rep1 != new_rep:
                        node_descriptions.pop(rep1, None)
                    if rep2 != new_rep:
                        node_descriptions.pop(rep2, None)

        # process rel types
        for rt in subset_rel_types[set_id]:
            rt_ds.add(rt)
            rel_num += 1

            if rel_num > 1 and random.random() < merge_prob:
                other = random.randint(0, rel_num - 1)
                other_rel = subset_rel_types[other // REL_TYPES_PER_UNIT][other % REL_TYPES_PER_UNIT]

                rep1 = rt_ds[rt]
                rep2 = rt_ds[other_rel]

                if rep1 != rep2:
                    rt_ds.merge(rep1, rep2)
                    new_rep = rt_ds[rep1]

                    merged_desc = merge_descriptions(
                        rel_descriptions[rep1], rel_descriptions[rep2]
                    )
                    rel_descriptions[new_rep] = merged_desc
                    if rep1 != new_rep:
                        rel_descriptions.pop(rep1, None)
                    if rep2 != new_rep:
                        rel_descriptions.pop(rep2, None)

        if (set_id + 1) % 100 == 0:
            print(f"Processed {set_id + 1}/{num_subsets} subsets...")

    t1 = time.time()

    # Step 3: remap triples
    remap_start = time.time()
    remapped_triples = [(ds[src], rt_ds[rel], ds[trg]) for (src, rel, trg) in triples]
    remap_end = time.time()

    # Results
    print("Final #node components:", ds.n_subsets)
    print("Final #rel components:", rt_ds.n_subsets)

    print(f"\nTiming:")
    print(f"  Build + merge DS: {t1 - t0:.2f} seconds")
    print(f"  Remap triples:    {remap_end - remap_start:.2f} seconds")

if __name__ == "__main__":
    main()
