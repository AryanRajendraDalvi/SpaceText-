import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from spacetext.rag import run_rag_agent

if __name__ == "__main__":
    print("🚀 SpaceText RAG Agent Test Interface 🚀")
    print("---------------------------------------")
    print("Type your question below (or 'quit' to exit):")
    
    while True:
        try:
            query = input("\n> ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query.strip():
                continue
                
            run_rag_agent(query)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
