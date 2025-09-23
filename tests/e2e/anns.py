import random
import string
import uuid
import time
import numpy as np
import faiss
from scipy.cluster.hierarchy import DisjointSet

DEGREE = 25
REL_TYPES_PER_UNIT = 500
VEC_DIM = 300
K = 10          # top-k for ANN search
MERGE_PROB = 0.99

def random_description(length=8):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def merge_descriptions(desc1, desc2):
    return random.choice([desc1, desc2])

def make_vector():
    return np.random.randn(VEC_DIM).astype("float32")

def main():
    total_elements = 1_000_000   # smaller for demo; scale up as needed
    subset_size = 1000
    num_subsets = total_elements // subset_size

    # Node disjoint set + descriptions
    ds = DisjointSet()
    node_descriptions = {}

    # Relationship type disjoint set + descriptions
    rt_ds = DisjointSet()
    rel_descriptions = {}

    # HNSW Faiss indexes
    entity_index = faiss.IndexHNSWFlat(VEC_DIM, 32)
    entity_index.hnsw.efSearch = 64
    rel_index = faiss.IndexHNSWFlat(VEC_DIM, 32)
    rel_index.hnsw.efSearch = 64

    # Mappings: index order → IDs
    entity_ids = []
    rel_ids = []

    # Counters for vectors
    entity_added = 0
    entity_avoided = 0
    rel_added = 0
    rel_avoided = 0

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

    # Step 2: build DS and do merges (exclude vector gen time)
    t0 = time.time()
    index_time = 0.0
    query_time = 0.0

    for set_id, input_set in enumerate(input_sets):
        # process nodes
        for elem in input_set:
            ds.add(elem)
            vec = make_vector()

            if len(entity_ids) > 0:
                # query nearest neighbors
                q_start = time.time()
                D, I = entity_index.search(vec.reshape(1, -1), K)
                query_time += time.time() - q_start

                cand_id = entity_ids[I[0][0]]
                rep1, rep2 = ds[elem], ds[cand_id]

                if rep1 != rep2 and random.random() < MERGE_PROB:
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

                    # if new node became root, add its vector
                    if new_rep == elem:
                        i_start = time.time()
                        entity_index.add(vec.reshape(1, -1))
                        entity_ids.append(new_rep)
                        entity_added += 1
                        index_time += time.time() - i_start
                    else:
                        entity_avoided += 1
                else:
                    i_start = time.time()
                    entity_index.add(vec.reshape(1, -1))
                    entity_ids.append(elem)
                    entity_added += 1
                    index_time += time.time() - i_start
            else:
                i_start = time.time()
                entity_index.add(vec.reshape(1, -1))
                entity_ids.append(elem)
                entity_added += 1
                index_time += time.time() - i_start

        # process rel types
        for rt in subset_rel_types[set_id]:
            rt_ds.add(rt)
            vec = make_vector()

            if len(rel_ids) > 0:
                q_start = time.time()
                D, I = rel_index.search(vec.reshape(1, -1), K)
                query_time += time.time() - q_start

                cand_id = rel_ids[I[0][0]]
                rep1, rep2 = rt_ds[rt], rt_ds[cand_id]

                if rep1 != rep2 and random.random() < MERGE_PROB:
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

                    if new_rep == rt:
                        i_start = time.time()
                        rel_index.add(vec.reshape(1, -1))
                        rel_ids.append(new_rep)
                        rel_added += 1
                        index_time += time.time() - i_start
                    else:
                        rel_avoided += 1
                else:
                    i_start = time.time()
                    rel_index.add(vec.reshape(1, -1))
                    rel_ids.append(rt)
                    rel_added += 1
                    index_time += time.time() - i_start
            else:
                i_start = time.time()
                rel_index.add(vec.reshape(1, -1))
                rel_ids.append(rt)
                rel_added += 1
                index_time += time.time() - i_start

        if (set_id + 1) % 10 == 0:
            print(f"Processed {set_id + 1}/{num_subsets} subsets...")

    t1 = time.time()

    # Step 3: remap triples
    remap_start = time.time()
    remapped_triples = [(ds[src], rt_ds[rel], ds[trg]) for (src, rel, trg) in triples]
    remap_end = time.time()

    # Results
    print("Final #node components:", ds.n_subsets)
    print("Final #rel components:", rt_ds.n_subsets)
    print("Sample node descriptions:", list(node_descriptions.items())[:5])
    print("Sample rel descriptions:", list(rel_descriptions.items())[:5])
    print("Sample remapped triples:", remapped_triples[:5])

    print(f"\nTiming (excluding vector gen):")
    build_merge_query_and_index_time = t1 - t0
    build_and_merge_time = build_merge_query_and_index_time - index_time - query_time
    print(f"  Total build + merge DS: {build_and_merge_time:.2f} s")
    print(f"  Total index add time:   {index_time:.2f} s")
    print(f"  Total query time:       {query_time:.2f} s")
    print(f"  Triple remap time:      {remap_end - remap_start:.2f} s")

    print(f"\nIndex stats:")
    print(f"  Entity vectors added:   {entity_added}")
    print(f"  Entity vectors avoided: {entity_avoided}")
    print(f"  Rel vectors added:      {rel_added}")
    print(f"  Rel vectors avoided:    {rel_avoided}")

if __name__ == "__main__":
    main()
