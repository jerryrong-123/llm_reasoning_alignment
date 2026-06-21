# ChromaVectorStore Backend Test Report

## Summary

- chroma_persist_dir: `D:\llm_reasoning_alignment_server_restored\outputs\hierarchical_rag\chroma\chroma_child_store`
- collection_name: `hierarchical_rag_child_chunks`
- collection_count: `944`
- query_index_id: `0`
- expected_child_id: `chunk_000001_000_000`
- top1_child_id: `chunk_000001_000_000`
- top1_matches_expected: `True`
- result_type_check: `True`
- returned_count: `5`
- latency_ms: `645.27`
- overall_pass: `True`

## Top results

### Rank 1

- child_id: `chunk_000001_000_000`
- parent_id: `parent_000001_000`
- title: `Radio City (Indian radio station)`
- score: `1.0`
- index_id: `0`

```text
Radio City is India's first private FM radio station and was started on 3 July 2001. It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2...
```

### Rank 2

- child_id: `chunk_000001_000_001`
- parent_id: `parent_000001_000`
- title: `Radio City (Indian radio station)`
- score: `0.7334737992406587`
- index_id: `1`

```text
It plays Hindi, English and regional songs. It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007. Radio City recently forayed into New Media in May 2008 with the launch...
```

### Rank 3

- child_id: `chunk_000001_000_002`
- parent_id: `parent_000001_000`
- title: `Radio City (Indian radio station)`
- score: `0.6714425976797929`
- index_id: `2`

```text
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features. The Radio station currently...
```

### Rank 4

- child_id: `chunk_000033_000_000`
- parent_id: `parent_000033_000`
- title: `List of magazines in Malaysia`
- score: `0.5569841533956507`
- index_id: `637`

```text
The first women's magazine was published in Malaysia in 1932. In the 2000s there were nearly fifty local titles addressing women in the country. These magazines also include those having an Islamic perspective.
```

### Rank 5

- child_id: `chunk_000007_000_000`
- parent_id: `parent_000007_000`
- title: `India`
- score: `0.556789675101129`
- index_id: `136`

```text
India, officially the Republic of India ("Bhārat Gaṇarājya"), is a country in South Asia. It is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous...
```

## Notes

This test verifies the ChromaVectorStore backend wrapper instead of calling chromadb directly.
It uses search_by_embedding(...) and does not load the BGE embedding model.
The goal is to confirm that ChromaVectorStore returns the unified VectorSearchResult structure expected by HierarchicalRetriever.