import garuddb
from garuddb import PersistentClient

client = PersistentClient(path='./aerospace_taxonomy.adk')
col = client.get_collection('aerospace_taxonomy')
query = 'How does a Decentralised Kalman Filter improve Autonomous Underwater Vehicle (AUV) navigation?'
res = col.query(query_texts=[query], n_results=5)
for doc, meta, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
    print(f"Dist: {dist:.4f} | Src: {meta.get('source')} | Doc: {doc[:100]}")
