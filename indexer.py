from typing import List, Optional
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import openai
from tqdm.auto import tqdm
import os
from lex_gpt.models import Record, TextRecord, Metadata

tokenizer = tiktoken.get_encoding('cl100k_base')


def tiktoken_len(text: str) -> int:
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


class Chunker:
    def __init__(self, chunk_size: Optional[int] = 400, chunk_overlap: Optional[int] = 20):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=tiktoken_len,
            separators=['\n\n', '.\n', '\n', '.', '?', '!', ' ', '']
        )

    def __call__(self, text_record: TextRecord) -> List[Record]:
        text_chunks = self.text_splitter.split_text(text_record.contenido)
        return [
            Record(
                id=f'{text_record.titulo}-{i}',
                text=text,
                metadata=Metadata(
                    titulo=text_record.titulo,
                    area=text_record.area,
                    tipo_de_texto=text_record.tipo_de_texto,
                    chunk=i
                )
            )
            for i, text in enumerate(text_chunks)
        ]


class Indexer:
    dimension_map = {
        'text-embedding-ada-002': 1536
    }

    def __init__(
            self, openai_api_key: Optional[str], pinecone_api_key: Optional[str],
            pinecone_environment: Optional[str], index_name: Optional[str] = "lex-gpt",
            embedding_model_name: Optional[str] = "text-embedding-ada-002",
            chunk_size: Optional[int] = 400, chunk_overlap: Optional[int] = 20
    ):
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        self.pinecone_api_key = pinecone_api_key or os.environ.get('PINECONE_API_KEY')
        self.pinecone_environment = pinecone_environment or os.environ.get('PINECONE_ENVIRONMENT')
        if self.openai_api_key is None:
            raise ValueError('openai_api_key not specified')
        if self.pinecone_api_key is None:
            raise ValueError('pinecone_api_key not specified')
        if self.pinecone_environment is None:
            raise ValueError('pinecone_environment not specified')

        self.chunker = Chunker(chunk_size, chunk_overlap)
        self.embedding_model_name = embedding_model_name
        self.metadata_config = {'indexed': list(Metadata.schema()['properties'].keys())}
        pinecone.init(
            api_key=pinecone_api_key, environment=pinecone_environment
        )
        openai.api_key = openai_api_key
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name, dimension=self.dimension_map[embedding_model_name],
                metadata_config=self.metadata_config
            )
        self.index = pinecone.GRPCIndex(index_name)

   def _embed(self, texts: List[str]) -> List[List[float]]:
    res = openai.Embedding.create(
        input=texts,
        engine=self.embedding_model_name
    )
    return [result["embedding"] for result in res["data"]]

def _index(self, records: List[Record]) -> None:
    ids = [record.id for record in records]
    texts = [record.text for record in records]
    metadatas = [dict(record.metadata) for record in records]
    for i, metadata in enumerate(metadatas):
        metadata['text'] = texts[i]
    embeddings = self._embed(texts)
    self.index.upsert(vectors=zip(ids, embeddings, metadatas))

def __call__(self, text_record: TextRecord, batch_size: Optional[int] = 100) -> None:
    chunks = self.chunker(text_record)
    for i in range(0, len(chunks), batch_size):
        i_end = min(i + batch_size, len(chunks))
        self._index(chunks[i:i_end])

