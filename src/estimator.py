import gc

import pandas as pd

from langchain.document_loaders import CSVLoader, DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    Doc,
    DatesExtractor,
)

class Normalize:
  def __init__(self):
    self.segmenter = Segmenter()
    self.morph_vocab = MorphVocab()
    self.emb = NewsEmbedding()
    self.morph_tagger = NewsMorphTagger(self.emb)
    self.syntax_parser = NewsSyntaxParser(self.emb)
    self.ner_tagger = NewsNERTagger(self.emb)
    self.names_extractor = NamesExtractor(self.morph_vocab)

  def normalize(self, doc):
    doc = Doc(doc)
    doc.segment(self.segmenter)
    doc.tag_morph(self.morph_tagger)
    doc.parse_syntax(self.syntax_parser)
    doc.tag_ner(self.ner_tagger)

    for token in doc.tokens:
        token.lemmatize(self.morph_vocab)

    for span in doc.spans:
        span.normalize(self.morph_vocab)
    res = [token.lemma for token in doc.tokens]

    return " ".join(res)

class BM25RetrieverWithScores(BM25Retriever):
    def get_relevant_documents_with_scores(self, query, k, score_threshold=0):
        processed_query = self.preprocess_func(query)
        scores = self.vectorizer.get_scores(processed_query)
        docs_and_scores = [(doc, score) for doc, score in zip(self.docs, scores)]
        sorted_docs_and_scores = sorted(
            docs_and_scores, key=lambda x: x[1], reverse=True
        )
        sorted_docs_and_scores = [
            i for i in sorted_docs_and_scores if i[1] > score_threshold
        ]
        res = []
        for doc, score in sorted_docs_and_scores[:k]:
            doc.metadata["score"] = score
            doc.metadata["method"] = "bm25"
            res.append(doc)

        return res


class SearchEngine:
    def __init__(self, file_path, model_name, normalizer):
        self.normalizer = normalizer
        print("Загружаем текстовые данные")
        documents = self._read_file(file_path)
        print("Зашружаем эмбеддинг модель")
        model = self._load_embedding_model(model_name)

        self.vector_db = self._create_vector_db(documents, model)
        del model
        gc.collect()

        self.bm25_db = self._create_bm25_db(documents)

    def _load_embedding_model(self, model_name):
        embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={"device": "cpu"}
        )
        return embeddings_model

    def _read_file(self, file_path):
        df_text = pd.read_csv(file_path)
        documents = DataFrameLoader(df_text, page_content_column="text").load()
        return documents

    def _create_vector_db(self, documents, embeddings_model):
        vector_db = FAISS.from_documents(documents, embeddings_model)
        return vector_db

    def _create_bm25_db(self, documents):
        bm25_db = BM25RetrieverWithScores.from_documents(
            documents,
            preprocess_func=lambda x: self.normalizer(x).split(),
        )
        return bm25_db

    def _retriever_vector(self, query, k):
        res = []
        for doc, score in self.vector_db.similarity_search_with_relevance_scores(
            query, k=k
        ):
            doc.metadata["method"] = "vector"
            doc.metadata["score"] = score
            res.append(doc)
        return res

    def _retriever_bm25(self, query, k):
        return self.bm25_db.get_relevant_documents_with_scores(query, k)

    def _retriever_comb(self, query, k):
        vect = self._retriever_vector(query, k)
        bm25 = self._retriever_bm25(query, k)
        res = []
        for index in range(max(len(vect), len(bm25))):
            if index < len(bm25):
                res.append(bm25[index])
            if index < len(vect):
                res.append(vect[index])

        res_new = []
        d_set = set()

        for doc in res:
            if doc.page_content not in d_set:
                res_new.append(doc)
                d_set.add(doc.page_content)

        return res[:k]

    def find_doc(self, query, search_engine, k):
        if search_engine == "vector":
            res = self._retriever_vector(query, k)

        if search_engine == "bm25":
            res = self._retriever_bm25(self.normalizer(query), k)

        if search_engine == "comb":
            res = self._retriever_comb(query, k)

        return [
            {
                "text": i.page_content,
                "score": round(i.metadata["score"], 2),
                "method": i.metadata["method"],
            }
            for i in res
        ]
