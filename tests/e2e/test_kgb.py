import asyncio
import neo4j
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as Embeddings
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm import VertexAILLM
from vertexai.generative_models import GenerationConfig
from concurrent.futures import ProcessPoolExecutor
import time

# --- move this out of test_kgb.py:test ---
def work_sync(file_path):
    # Important: re-import inside the worker if needed to avoid state sharing
    import asyncio
    from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
    from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
    from neo4j_graphrag.llm import OpenAILLM as LLM
    from neo4j_graphrag.experimental.components.entity_relation_extractor import \
        LLMEntityRelationExtractor
    pdf_loader = PdfLoader()
    splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=200)
    openai_llm = LLM(
        model_name="gpt-4o-mini",
        model_params={
            "response_format": {"type": "json_object"},
            "temperature": 0
        })

    entity_extractor = LLMEntityRelationExtractor(
        llm=openai_llm,
        prompt_template=ERExtractionTemplate.DEFAULT_TEMPLATE,
    )

    pdf_document = asyncio.run(pdf_loader.run(filepath=file_path))
    text_chunks = asyncio.run(splitter.run(text=pdf_document.text))
    graph = asyncio.run(entity_extractor.run(chunks=text_chunks))
    return graph



def test(driver: neo4j.Driver):
  neo4j_driver = driver
  #NEO4J_URI = os.getenv("NEO4J_URI")
  ## NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
  #NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
  #NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

  # neo4j_driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

  embedder = Embeddings()

  # 1. Build KG and Store in Neo4j Database
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


  # kg_builder_pdf = SimpleKGPipeline(
  #    llm=ex_llm,
  #    driver=neo4j_driver,
  #    embedder=embedder,
  #    from_pdf=True,
  #    perform_entity_resolution=False,
  # )
  # async def work(file_path):
  #     pdf_document = await pdf_loader.run(filepath=file_path)
  #     text_chunks = await splitter.run(text=pdf_document.text)
  #
  # async def process_doc(file_path):
  #   import time
  #   semaphore = asyncio.Semaphore(10)  # pool size = 10
  #
  #   async def pooled_work():
  #       async with semaphore:
  #           return await work(file_path)
  #
  #   tasks = [pooled_work() for i in range(100)]
  #   t0 = time.time()
  #   results = await asyncio.gather(*tasks)
  #   elapsed = time.time() - t0
  #   print(20*"woefijweofjiwef")
  #   print(elapsed)
  #   #graph = await entity_extractor.run(chunks=text_chunks)
  #   #return graph
  #   return 0


  async def process_doc(file_path):
      t0 = time.time()

      loop = asyncio.get_running_loop()

      with ProcessPoolExecutor(max_workers=1) as pool:
          tasks = [loop.run_in_executor(pool, work_sync, file_path) for _ in range(1)]
          results = await asyncio.gather(*tasks)

      elapsed = time.time() - t0
      print(20 * "woefijweofjiwef")
      print(f"Elapsed: {elapsed:.2f}s")
      print(f"Got {len(results)} results")

      return results


  #
  # # 2. KG Retriever
  # vector_retriever = VectorRetriever(
  #    neo4j_driver,
  #    index_name="text_embeddings",
  #    embedder=embedder
  # )
  #
  # # 3. GraphRAG Class
  # llm = LLM(model_name="gpt-4o")
  # rag = GraphRAG(llm=llm, retriever=vector_retriever)
  #
  # # 4. Run
  # response = rag.search( "How is precision medicine applied to Lupus?")
  # print(response.answer)

  graph = asyncio.run(process_doc("test.pdf"))
  print("foo")
  #print(len(graph.nodes))
