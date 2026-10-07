import sys
try:
    from garuddb import PersistentClient
    client = PersistentClient(path='./aerospace_taxonomy.adk')
    col = client.get_collection('aerospace_taxonomy')
    
    sources = ['isro', 'nasa', 'drdo', 'esa', 'arxiv', 'openalex', 'nasa_news']
    for src in sources:
        res = col.get(where={'source': src})
        print(f'{src}: {len(res.get("ids", []))}')
except Exception as e:
    print(f"Error: {e}")
