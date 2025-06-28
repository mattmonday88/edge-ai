from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

def index_document(doc_id, text, summary, entities, layout):
    body = {
        "text": text,
        "summary": summary,
        "entities": entities,
        "layout": [region.to_dict() for region in layout],
    }
    es.index(index="documents", id=doc_id, body=body)