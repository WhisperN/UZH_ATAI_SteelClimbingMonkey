import torch, pickle, hashlib
from pathlib import Path
from rdflib import Graph, Namespace
from sentence_transformers import SentenceTransformer, util


class OcciglotSPARQL:
    def __init__(self):
        print("Initializing Occiglot SPARQL model with dataset context...\n")

        self.DATASET_URL = "https://files.ifi.uzh.ch/ddis/teaching/2025/ATAI/dataset/graph.nt"
        self.CACHE_DIR = Path("occiglot_cache")
        self.CACHE_DIR.mkdir(exist_ok=True)

        self.dataset_hash = hashlib.md5(self.DATASET_URL.encode()).hexdigest()
        self.GRAPH_CACHE = self.CACHE_DIR / f"graph_{self.dataset_hash}.pkl"
        self.ENTITY_EMB_CACHE = self.CACHE_DIR / f"entity_emb_{self.dataset_hash}.pt"
        self.PRED_EMB_CACHE = self.CACHE_DIR / f"pred_emb_{self.dataset_hash}.pt"
        self.ENTITY_LABELS_CACHE = self.CACHE_DIR / f"entity_labels_{self.dataset_hash}.pkl"
        self.PREDICATES_CACHE = self.CACHE_DIR / f"predicates_{self.dataset_hash}.pkl"

        if self.GRAPH_CACHE.exists():
            print("→ Loading cached RDF graph...")
            with open(self.GRAPH_CACHE, "rb") as f:
                self.graph = pickle.load(f)
        else:
            print("→ Parsing RDF graph (first-time load, please wait)...")
            g = Graph()
            g.parse(self.DATASET_URL, format="nt")
            with open(self.GRAPH_CACHE, "wb") as f:
                pickle.dump(g, f)
            self.graph = g

        print(f"Loaded {len(self.graph)} triples.")

        g = self.graph
        g.bind("wd", "http://www.wikidata.org/entity/")
        g.bind("wdt", "http://www.wikidata.org/prop/direct/")
        g.bind("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        g.bind("schema", "http://schema.org/")
        g.bind("ddis", "http://ddis.ch/atai/")

        self.RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        self.prefix_block = "\n".join([
            "PREFIX wd: <http://www.wikidata.org/entity/>",
            "PREFIX wdt: <http://www.wikidata.org/prop/direct/>",
            "PREFIX p: <http://www.wikidata.org/prop/>",
            "PREFIX ps: <http://www.wikidata.org/prop/statement/>",
            "PREFIX pq: <http://www.wikidata.org/prop/qualifier/>",
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "PREFIX ddis: <http://ddis.ch/atai/>",
            "PREFIX schema: <http://schema.org/>"
        ])

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        if self.PREDICATES_CACHE.exists() and self.ENTITY_LABELS_CACHE.exists():
            print("→ Loading cached entity labels and predicates...")
            self.predicates = pickle.load(open(self.PREDICATES_CACHE, "rb"))
            self.entities = pickle.load(open(self.ENTITY_LABELS_CACHE, "rb"))
        else:
            print("→ Extracting entities and predicates (first-time preprocessing)...")
            self.predicates = sorted(set(str(p) for _, p, _ in g))
            self.entities = [
                (str(s), str(o)) for s, p, o in g.triples((None, self.RDFS.label, None))
                if isinstance(o, str)
            ]
            pickle.dump(self.predicates, open(self.PREDICATES_CACHE, "wb"))
            pickle.dump(self.entities, open(self.ENTITY_LABELS_CACHE, "wb"))

        self.entity_texts = [label for _, label in self.entities]

        if self.PRED_EMB_CACHE.exists() and self.ENTITY_EMB_CACHE.exists():
            print("→ Loading cached embeddings...")
            self.pred_embeddings = torch.load(self.PRED_EMB_CACHE)
            self.entity_embeddings = torch.load(self.ENTITY_EMB_CACHE)
        else:
            print("→ Computing embeddings (first-time only, may take a minute)...")
            self.pred_embeddings = self.embedder.encode(
                self.predicates, convert_to_tensor=True,
                batch_size=64, show_progress_bar=True
            )
            self.entity_embeddings = self.embedder.encode(
                self.entity_texts, convert_to_tensor=True,
                batch_size=64, show_progress_bar=True
            )
            torch.save(self.pred_embeddings, self.PRED_EMB_CACHE)
            torch.save(self.entity_embeddings, self.ENTITY_EMB_CACHE)

        print("Model and KG context ready (cached loading will now be instant).\n")


    def _find_entities(self, question: str, top_k: int = 2):
        q_emb = self.embedder.encode(question, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, self.entity_embeddings).flatten()
        top = torch.topk(cos_scores, k=min(top_k, len(self.entities)))
        return sorted([(self.entities[i][0], self.entities[i][1], float(cos_scores[i])) for i in top.indices], key=lambda t: t[2])

    def _find_predicates(self, question: str, top_k: int = 3):
        q_emb = self.embedder.encode(question, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, self.pred_embeddings).flatten()
        top = torch.topk(cos_scores, k=min(top_k, len(self.predicates)))
        return [(self.predicates[i], float(cos_scores[i])) for i in top.indices]


    def _compose_single_entity_query(self, entity_uri: str, predicates: list):
        vars_ = [f"?v{i}" for i in range(len(predicates))]
        body = " .\n  ".join([f"<{entity_uri}> <{pred}> {vars_[i]} " for i, pred in enumerate(predicates)])
        return f"""{self.prefix_block}
SELECT {' '.join(vars_)} WHERE {{
  {body}.
}}"""

    def _compose_multi_entity_query(self, entities: list, predicates: list):
        n = len(entities)
        vars_ = [f"?x{i}" for i in range(n - 1)]
        lines = [f"<{entities[0][0]}> ?p1 {vars_[0]} ."]
        for i in range(1, n - 1):
            prev_var = vars_[max(0, i - 1)]
            lines.append(f"{prev_var} ?p{i+1} <{entities[i][0]}> .")
        if n > 2:
            last_var = vars_[-1]
            lines.append(f"{last_var} ?p{n} <{entities[-1][0]}> .")
        body = "\n  ".join(lines)
        select_vars = " ".join(vars_)
        return f"""{self.prefix_block}
SELECT DISTINCT {select_vars} WHERE {{
  {body}
}}"""

    def ask(self, question: str):
        entities = self._find_entities(question)
        predicates = self._find_predicates(question)
        print(f"Detected entities: {[e[1] for e in entities]}")
        print(f"Top predicates: {[p[0].split('/')[-1] for p in predicates]}")

        if len(entities) >= 2:
            print('several entities detected')
            return self._compose_multi_entity_query(entities, predicates)
        elif len(entities) == 1 and len(predicates) >= 1:
            print('one entity detected')
            return self._compose_single_entity_query(entities[0][0], [p[0] for p in predicates])
        else:
            return ""
