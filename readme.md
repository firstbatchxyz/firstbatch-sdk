# FirstBatch SDK

The FirstBatch SDK provides an interface for integrating vector databases and powering personalized AI experiences in your application.

## Key Features

- Seamlessly manage user sessions with persistent IDs or temporary sessions
- Send signal actions like likes, clicks, etc. to update user embeddings in real-time
- Fetch personalized batches of data tailored to each user's embeddings  
- Support for multiple vector database integrations: Pinecone, Weaviate, etc.
- Built-in algorithms for common personalization use cases
- Easy configuration with Python classes and environment variables

## Getting Started

### Prerequisites

- Python 3.9+
- API keys for FirstBatch and your chosen vector database

### Installation

```
pip install firstbatch
```

## Basic Usage

1. **Initialize VectorDB of your choice**
    ```python
   api_key = os.environ["PINECONE_API_KEY"]
   env = os.environ["PINECONE_ENV"]

   pinecone.init(api_key=api_key, environment=env)
   index = pinecone.Index("your_index_name")
   
   # Init FirstBatch
   config = Config(batch_size=20)
   personalized = FirstBatch(api_key=os.environ["FIRSTBATCH_API_KEY"], config=config)
   
   personalized.add_vdb("my_db", Pinecone(index, embedding_size=1536))
    ```

### Personalization

2. **Create a session with an Algorithm suiting your needs**
    ```python 
   session = personalized.session(algorithm=AlgorithmLabel.AI_AGENTS, vdbid="my_db")
    ```

3. **Make recommendations**
    ```python
   ids, batch = personalized.batch(session)
    ```
4. **Let users add signals to shape their embeddings**
   ```python
   user_pick = 0  # User liked the first content from the previous batch.
   personalized.add_signal(session, UserAction(Signal.LIKE), ids[user_pick])
   ```

## Support

For any issues or queries contact `support@firstbatch.xyz`.

  
## Resources

- [User Embedding Guide](https://firstbatch.gitbook.io/user-embeddings/)
- [SDK Documentation](https://firstbatch.gitbook.io/firstbatch-sdk/)

Feel free to dive into the technicalities and leverage FirstBatch SDK for highly personalized user experiences.
