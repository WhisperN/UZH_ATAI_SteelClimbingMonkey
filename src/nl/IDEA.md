## Pipeline for nl processing

### Requirements

- Factual Questions
- Embedding Questions

#### Factual example
Q: Who is the director of Good Will Hunting?

A: Gus Van Sant is the director of Good Will Hunting.

#### Embedding example
Q: Who is the screenwriter of The Masked Gang: Cyprus?

A: The answer suggested by embeddings: Cengiz Küçükayvaz, Murat Aslan, and Melih Ekener.

### Pipeline

1. Predict Question type
2. Convert to SPARQL
3. Parse Response
4. Convert to NL