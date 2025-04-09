import asyncio
import random
from typing import Dict, List, Any
from dataclasses import dataclass
from uuid import uuid4
from qdrant_client import models
import qdrant_client
import asyncio
from qdrant_utils import async_get_point_by_id, async_get_random_point

@dataclass
class VectorSearchResult:
    """Simulated vector search result"""
    id: str
    score: float
    payload: Dict[str, Any]


class FakeQdrantClient:
    """
    A fake asynchronous Qdrant client that simulates vector database operations
    """
    
    def __init__(
        self, 
        collection_name: str = "test_collection", 
        delay_range: tuple = (0.1, 0.5)
    ):
        self.collection_name = collection_name
        self.delay_range = delay_range
        print(
            f"Initialised FakeQdrantClient for collection '{collection_name}'"
        )
    
    async def simulate_delay(self):
        """Simulate network delay"""
        delay = random.uniform(*self.delay_range)
        await asyncio.sleep(delay)
    
    async def search(
        self, query_vector: List[float], limit: int = 10
    ) -> List[VectorSearchResult]:
        """Simulate an asynchronous vector search operation"""
        await self.simulate_delay()
        
        # Generate fake results
        results = []
        for _ in range(min(limit, 5)):  # Limit to 5 max to keep things simple
            results.append(VectorSearchResult(
                id=str(uuid4()),
                score=random.uniform(0.5, 0.99),
                payload={"text": f"Sample text {random.randint(1, 100)}"}
            ))
        
        return sorted(results, key=lambda x: x.score, reverse=True)


class AsyncTaskHandler:
    """Handler for managing asynchronous search tasks with embeddings"""
    
    def __init__(self, collection_name:str, real_client: qdrant_client.AsyncQdrantClient, fake_client: FakeQdrantClient):
        self.real_client = real_client
        self.fake_client = fake_client
        self.collection_name = collection_name
        self.tasks = []
        self.results_by_id = {}
    
    async def search_with_embedding(
        self, 
        task_id: str, 
        embedding: List[float],
        limit: int = 5
    ) -> Dict[str, Any]:
        """Perform a search with the given embedding and return results"""
        try:
            # results = await self.fake_client.search(embedding, limit=limit)
            results = await async_get_random_point(self.real_client, self.collection_name)
            
            return {
                "task_id": task_id,
                "results": results[0].vector[0],
                "status": "success"
            }
        except Exception as e:
            print(f"Search error for task {task_id}: {e}")
            return {
                "task_id": task_id,
                "results": [],
                "status": "error",
                "error": str(e)
            }
    
    async def process_embeddings(
        self, embeddings: Dict[str, List[float]], limit: int = 5
    ) -> Dict[str, Any]:
        """
        Process multiple embeddings concurrently and collect results
        
        Args:
            embeddings: Dictionary mapping task_id to embedding vector
            limit: Maximum number of results per search
            
        Returns:
            Dictionary mapping task_id to search results
        """
        # Clear previous tasks
        self.tasks = []
        
        # Create tasks for all embeddings directly
        for task_id, embedding in embeddings.items():
            task = asyncio.create_task(
                self.search_with_embedding(task_id, embedding, limit)
            )
            self.tasks.append(task)
        
        if not self.tasks:
            print("No tasks to run")
            return {}
        
        # Wait for all tasks to complete and gather results
        all_results = await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Organize results by task_id
        results_by_id = {}
        for result in all_results:
            if isinstance(result, Exception):
                print(f"Task failed with error: {result}")
            else:
                results_by_id[result["task_id"]] = result
        
        self.results_by_id = results_by_id
        return results_by_id


async def main():
    """Main function demonstrating the async task handler with embeddings"""
    # Create a fake client
    client = FakeQdrantClient(collection_name="test_vectors")
    real_client = qdrant_client.AsyncQdrantClient("localhost")
    # Create task handler
    handler = AsyncTaskHandler("colours", real_client, client)
    
    # Create sample embeddings (each is a task with an ID and embedding vector)
    sample_ids = {(id, id):[str(id) for id in range(100)] for id in range(100)}
    
    # Process all embeddings concurrently
    print(f"Processing {len(sample_ids)} embeddings...")
    results = await handler.process_embeddings(sample_ids, limit=3)
    
    # Display results
    print("\nResults:")
    for task_id, result in results.items():
        print(f"\nTask ID: {task_id}")
        print(f"Status: {result['status']}")
        print(f" {result} result:")
        
        # for i, match in enumerate(result['results']):
        #     print(f"  Match {i+1}: ID={match.id}, Score={match.score:.3f}")
        #     print(f"    Payload: {match.payload}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())