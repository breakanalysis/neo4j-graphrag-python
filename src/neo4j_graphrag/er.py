import asyncio
import random
from collections import Counter, defaultdict

from neo4j_graphrag.experimental.components.types import Neo4jNode, Neo4jRelationship, \
    Neo4jGraph
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm import VertexAILLM
from vertexai.generative_models import GenerationConfig

def sample_document(num_labels=50, num_types_per_pair=35):
    entity_counts = Counter({'Concept': 107, 'Entity': 57, 'Threat': 25, 'Control': 20,
                             'Component': 10, 'Document': 9, 'Standard': 7, 'Platform': 7,
                             'Organization': 6, 'Tool': 6, 'Person': 2, 'Architecture Component': 2,
                             'Application': 2, 'Action': 2, 'Year': 1, 'Objective': 1, 'Framework': 1,
                             'Policy': 1, 'Data': 1, 'Event': 1, 'Device': 1, 'Service': 1}
                            )

    meta_triples = Counter(
        {('Concept', 'RELATED_TO', 'Concept'): 32,
         ('Control', 'MITIGATES', 'Threat'): 10, ('Entity', 'RELATED_TO', 'Entity'): 8,
         ('Threat', 'RELATED_TO', 'Threat'): 7,
         ('Concept', 'INCLUDES', 'Concept'): 7,
         ('Tool', 'USES', 'Tool'): 5, ('Platform', 'PROVIDES', 'Platform'): 5,
         ('Concept', 'PART_OF', 'Concept'): 5, ('Concept', 'MITIGATES', 'Concept'): 5,
         ('Policy', 'REFERS_TO', 'Standard'): 4,
         ('Document', 'REFERENCES', 'Component'): 4,
         ('Architecture Component', 'DESCRIBES', 'Control'): 4,
         ('Concept', 'RESULTS_IN', 'Concept'): 4,
         ('Concept', 'INTEGRATES', 'Concept'): 3, ('Concept', 'USES', 'Concept'): 3,
         ('Threat', 'MITIGATES', 'Component'): 3,
         ('Organization', 'USES', 'Concept'): 2, ('Concept', 'COMPRISES', 'Concept'): 2,
         ('Person', 'AUTHORED', 'Document'): 2,
         ('Entity', 'ACCESSIBLE_DATA_SOURCE', 'Entity'): 2,
         ('Concept', 'AFFECTS', 'Concept'): 2, ('Entity', 'GAINS_ACCESS', 'Entity'): 2,
         ('Entity', 'USES', 'Entity'): 2, ('Entity', 'LEAKS', 'Entity'): 2,
         ('Platform', 'INCLUDES', 'Concept'): 2,
         ('Component', 'HAS_CONTROL', 'Control'): 2,
         ('Document', 'REFERENCES', 'Control'): 2,
         ('Control', 'RELATED_TO', 'Component'): 2,
         ('Application', 'MITIGATES', 'Threat'): 2,
         ('Application', 'VALIDATES', 'Threat'): 2,
         ('Organization', 'OWNS', 'Document'): 1,
         ('Document', 'PUBLISHED_IN', 'Year'): 1,
         ('Organization', 'CREATED', 'Document'): 1,
         ('Document', 'COVERS', 'Concept'): 1, ('Document', 'AIMS_TO', 'Objective'): 1,
         ('Document', 'INCLUDES', 'Concept'): 1,
         ('Organization', 'ENCOURAGES', 'Concept'): 1,
         ('Organization', 'IDENTIFIES', 'Concept'): 1,
         ('Concept', 'APPLIES', 'Concept'): 1,
         ('Document', 'INTEGRATES', 'Framework'): 1,
         ('Concept', 'ORGANIZED_BY', 'Concept'): 1,
         ('Component', 'USES', 'Data'): 1,
         ('Component', 'DEPLOYS', 'Component'): 1,
         ('Component', 'ALTERNATIVE_TO', 'Component'): 1,
         ('Entity', 'DEPLOYS', 'Entity'): 1,
         ('Entity', 'ALLOWS', 'Entity'): 1, ('Entity', 'ALLEVATES', 'Entity'): 1,
         ('Entity', 'RESPONDS_TO', 'Entity'): 1, ('Entity', 'INVOKES', 'Entity'): 1,
         ('Entity', 'COMMUNICATES_THROUGH', 'Entity'): 1,
         ('Entity', 'WHITELISTED', 'Entity'): 1, ('Entity', 'OFFERED_AS', 'Entity'): 1,
         ('Entity', 'REFERS_TO', 'Entity'): 1, ('Concept', 'REQUIRES', 'Concept'): 1,
         ('Concept', 'IMPACT_ON', 'Concept'): 1, ('Entity', 'UPLOADS', 'Entity'): 1,
         ('Entity', 'DOWNLOADS', 'Entity'): 1, ('Entity', 'EXPOSES', 'Entity'): 1,
         ('Event', 'AFFECTS', 'Entity'): 1,
         ('Entity', 'MANIPULATES', 'Entity'): 1, ('Entity', 'INJECTS', 'Entity'): 1,
         ('Entity', 'EXECUTES', 'Entity'): 1, ('Entity', 'PROVIDES', 'Entity'): 1,
         ('Entity', 'ACCIDENTALLY_LEAKS', 'Entity'): 1,
         ('Entity', 'COLLECTS', 'Entity'): 1, ('Entity', 'RESEARCHES', 'Entity'): 1,
         ('Entity', 'REPLICATES', 'Entity'): 1, ('Entity', 'DISRUPTS', 'Entity'): 1,
         ('Entity', 'ATTACKS', 'Entity'): 1,
         ('Control', 'APPLIES_TO', 'Architecture Component'): 1,
         ('Organization', 'IMPLEMENTS', 'Device'): 1,
         ('Organization', 'REQUIRES', 'Concept'): 1,
         ('Concept', 'SECURES', 'Concept'): 1, ('Device', 'USES', 'Concept'): 1,
         ('Concept', 'ISSUES', 'Device'): 1,
         ('Platform', 'IMPLEMENTATION_OF', 'Organization'): 1,
         ('Platform', 'PROVIDES', 'Service'): 1,
         ('Component', 'CONTAINS', 'Control'): 1,
         ('Platform', 'PROVIDES_STANDARDIZED_APPROACH', 'Component'): 1,
         ('Control', 'RELATED_TO', 'Threat'): 1,
         ('Control', 'VERIFIES', 'Component'): 1, ('Control', 'USES', 'Component'): 1,
         ('Threat', 'RELATED_TO', 'Component'): 1, ('Entity', 'POISONING', 'Entity'): 1,
         ('Entity', 'DATA_LEAKAGE', 'Entity'): 1,
         ('Entity', 'POISONING_THEFT', 'Entity'): 1,
         ('Entity', 'EVALUATES', 'Entity'): 1,
         ('Entity', 'ARCHITECTURE_COMPONENT', 'Entity'): 1,
         ('Concept', 'REFERENCES', 'Concept'): 1,
         ('Organization', 'IMPLEMENTS', 'Concept'): 1,
         ('Concept', 'MANAGES', 'Concept'): 1, ('Concept', 'ENCRYPTS', 'Concept'): 1,
         ('Document', 'HAS_COMPONENT', 'Architecture Component'): 1,
         ('Application', 'VALIDATES', 'Application'): 1,
         ('Concept', 'RELATED_TO', 'Action'): 1,
         ('Concept', 'RELATED_TO', 'Standard'): 1,
         ('Component', 'RELATED_TO', 'Action'): 1,
         ('Standard', 'REFERENCES', 'Concept'): 1,
         ('Concept', 'VALIDATES', 'Concept'): 1, ('Entity', 'VALIDATES', 'Entity'): 1,
         ('Entity', 'VALIDATES', 'Concept'): 1,
         ('Concept', 'REFERENCES', 'Standard'): 1}
    )

    # === 1. Unique label assignment ===
    entity_types = list(entity_counts.keys())
    if len(entity_types) > num_labels:
        raise ValueError("Not enough num_labels to assign unique labels to all entity types")

    available_labels = [f"Entity_{i}" for i in range(1, num_labels + 1)]
    random.shuffle(available_labels)
    label_map = {etype: available_labels.pop() for etype in entity_types}

    # === 2. Nodes ===
    nodes = []
    node_registry = {}
    for idx, etype in enumerate(entity_types):
        node = Neo4jNode(
            id=str(idx),
            label=label_map[etype],
            properties={"original_type": etype}
        )
        nodes.append(node)
        node_registry[etype] = node.id

    # === 3. Count how many unique relationship *types* needed per (src, dst) ===
    grouped = defaultdict(int)
    for (src, _, dst) in meta_triples.keys():
        grouped[(src, dst)] += 1

    # === 4. Assign suffix pools per (src, dst) ===
    suffix_pools = {}
    for (src, dst), need in grouped.items():
        if need > num_types_per_pair:
            raise ValueError(
                f"Need {need} distinct types for ({src}, {dst}), "
                f"but only {num_types_per_pair} available"
            )
        suffix_pools[(src, dst)] = random.sample(range(1, num_types_per_pair + 1), need)

    # === 5. Map each original triple to a unique relationship type ===
    relationship_map = {}
    for (src, rel, dst) in meta_triples.keys():
        suffix = suffix_pools[(src, dst)].pop()  # consume one unique suffix
        rel_type = f"{label_map[src]}_TO_{label_map[dst]}_{suffix}"
        relationship_map[(src, rel, dst)] = rel_type

    # === 6. Generate relationships ===
    relationships = []
    for (src, rel, dst), count in meta_triples.items():
        relationships.append(
            Neo4jRelationship(
                start_node_id=node_registry[src],
                end_node_id=node_registry[dst],
                type=relationship_map[(src, rel, dst)]
            )
        )

    return Neo4jGraph(nodes=nodes, relationships=relationships)


def extract(input_file):
    pdf_loader = PdfLoader()
    splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=200)

    gemini_llm = VertexAILLM(
        model_name="gemini-2.0-flash-001",
        generation_config=(GenerationConfig(temperature=1.0)),
        # add here any argument that will be passed to the
        # vertexai.generative_models.GenerativeModel client
    )
    openai_llm=LLM(
        model_name="gpt-4o-mini",
        model_params={
            "response_format": {"type": "json_object"},
            "temperature": 0
        })

    entity_extractor = LLMEntityRelationExtractor(
        llm=openai_llm,
        prompt_template=ERExtractionTemplate.DEFAULT_TEMPLATE,
    )

    async def process_doc(file_path):
        pdf_document = await pdf_loader.run(filepath=file_path)
        text_chunks = await splitter.run(text=pdf_document.text)
        graph = await entity_extractor.run(chunks=text_chunks)
        return graph

    return asyncio.run(process_doc(input_file))
