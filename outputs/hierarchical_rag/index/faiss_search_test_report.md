# FAISS Search Test Report

## Summary

- Index: `outputs/hierarchical_rag/index/faiss_child.index`
- Metadata: `outputs/hierarchical_rag/index/faiss_child_meta.json`
- Embedding model: `BAAI/bge-small-en-v1.5`
- Query embedding: `sentence-transformers`
- Search index type: `IndexFlatIP`

## Test Queries

### Query: Which magazine was started first Arthur's Magazine or First for Women?

- Latency ms: 303.62

1. score=0.7517, child_id=chunk_000033_001_000, parent_id=parent_000033_001, title=First for Women

   First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey.

2. score=0.7517, child_id=chunk_000001_007_000, parent_id=parent_000001_007, title=First for Women

   First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey.

3. score=0.7513, child_id=chunk_000001_005_000, parent_id=parent_000001_005, title=Arthur's Magazine

   Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged...

4. score=0.6883, child_id=chunk_000033_009_000, parent_id=parent_000033_009, title=List of magazines in China

   In 1898 the first women's magazine was published in China. The number of women's magazines has increased in the country since the late 1980s. In addition to national titles international magazines are also published in the country. "

5. score=0.6667, child_id=chunk_000033_000_000, parent_id=parent_000033_000, title=List of magazines in Malaysia

   The first women's magazine was published in Malaysia in 1932. In the 2000s there were nearly fifty local titles addressing women in the country. These magazines also include those having an Islamic perspective.

### Query: Arthur's Magazine start date

- Latency ms: 14.77

1. score=0.7135, child_id=chunk_000001_005_000, parent_id=parent_000001_005, title=Arthur's Magazine

   Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged...

2. score=0.6191, child_id=chunk_000003_005_003, parent_id=parent_000003_005, title=Los Angeles Reader

   It is famous for being the first newspaper to publish Matt Groening's cartoon strip, Life in Hell on April 25, 1980. James Vowell hired Matt Groening as his assistant editor in 1979. Groening was also originally a Reader music critic.

3. score=0.5904, child_id=chunk_000033_001_000, parent_id=parent_000033_001, title=First for Women

   First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey.

4. score=0.5904, child_id=chunk_000001_007_000, parent_id=parent_000001_007, title=First for Women

   First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey.

5. score=0.5879, child_id=chunk_000033_009_000, parent_id=parent_000033_009, title=List of magazines in China

   In 1898 the first women's magazine was published in China. The number of women's magazines has increased in the country since the late 1980s. In addition to national titles international magazines are also published in the country. "

### Query: First for Women start date

- Latency ms: 9.41

1. score=0.6948, child_id=chunk_000033_001_000, parent_id=parent_000033_001, title=First for Women

   First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey.

2. score=0.6948, child_id=chunk_000001_007_000, parent_id=parent_000001_007, title=First for Women

   First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey.

3. score=0.5977, child_id=chunk_000033_000_000, parent_id=parent_000033_000, title=List of magazines in Malaysia

   The first women's magazine was published in Malaysia in 1932. In the 2000s there were nearly fifty local titles addressing women in the country. These magazines also include those having an Islamic perspective.

4. score=0.5920, child_id=chunk_000033_009_000, parent_id=parent_000033_009, title=List of magazines in China

   In 1898 the first women's magazine was published in China. The number of women's magazines has increased in the country since the late 1980s. In addition to national titles international magazines are also published in the country. "

5. score=0.5829, child_id=chunk_000033_007_001, parent_id=parent_000033_007, title=My Secret Garden

   Later, after other women began writing and talking about sex publicly, Friday began thinking about writing a book about female sexual fantasies, first collecting fantasies from her friends, and then advertising in newspapers and magazines for more. She organiz...

### Query: Radio City Indian radio station

- Latency ms: 8.26

1. score=0.8344, child_id=chunk_000001_000_000, parent_id=parent_000001_000, title=Radio City (Indian radio station)

   Radio City is India's first private FM radio station and was started on 3 July 2001. It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003)...

2. score=0.7914, child_id=chunk_000001_000_001, parent_id=parent_000001_000, title=Radio City (Indian radio station)

   It plays Hindi, English and regional songs. It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com...

3. score=0.7693, child_id=chunk_000001_000_002, parent_id=parent_000001_000, title=Radio City (Indian radio station)

   Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features. The Radio station currently plays a mix of Hindi and Regional music....

4. score=0.5598, child_id=chunk_000030_009_000, parent_id=parent_000030_009, title=Tinley Park station

   Tinley Park Station (also known as Tinley Park-Oak Park Avenue Station) is an elaborate commuter railroad station along Metra's Rock Island District line in Tinley Park, Illinois. The station is officially located at 6700 South Street between Oak Park Avenue a...

5. score=0.5483, child_id=chunk_000019_008_001, parent_id=parent_000019_008, title=New Rules (song)

   It was released to contemporary hit radio in the United Kingdom on 21 July 2017 as the album's sixth single. It impacted the same format in the United States on 22 August 2017.
