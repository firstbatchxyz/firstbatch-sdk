# FirstBatch SDK

## Getting Started

1. **Installation**
    ```bash
    pip install firstbatch-sdk
    ```

2. **Initialize VectorDB of your choice**
    ```python
   api_key = os.environ["PINECONE_API_KEY"]
   env = os.environ["PINECONE_ENV"]

   pinecone.init(api_key=api_key, environment=env)
   index = pinecone.Index("your_index_name")
    ```

## Personalization

1. **Create a session with an Algorithm suiting your needs**
    ```python 
   session = personalized.session(algorithm=AlgorithmLabel.AI_AGENTS, vdbid="my_pinecone_db")
    ```

2. **Make recommendations**
    ```python
   session_id = session.data
   ids, batch = personalized.batch(session_id)
    ```
3. **Let users add signals to shape their embeddings**
   ```python
   user_pick = 0  # User liked the first content from the previous batch.
   personalized.add_signal(session_id, UserAction(Signal.LIKE), ids[user_pick])
   ```

## API Reference

For full API documentation, visit [API Docs](your_documentation_url_here).

## Support

For any issues or queries, refer to our [Support Page](your_support_page_url_here) or contact `support@firstbatch.com`.

## License

FirstBatch SDK is licensed under [MIT License](your_license_url_here).
  
## Resources

- [User Embedding Guide](your_guide_url_here)
- [API Documentation](your_documentation_url_here)

Feel free to dive into the technicalities and leverage FirstBatch SDK for highly personalized user experiences.
