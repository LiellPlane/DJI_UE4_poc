import asyncio
import random
from typing import Dict, List, Any
from dataclasses import dataclass
from uuid import uuid4
import sys
import pathlib
# Add all parent directories to Python path
current_path = pathlib.Path(__file__).parent.absolute()
parent_path = current_path.parent
sys.path.insert(0, str(parent_path))

from qdrant_client import models
import qdrant_client
import asyncio
from qdrant_utils import async_get_closest_match, async_get_point_by_id, async_get_random_point, get_embedding_average, async_get_embedding_average

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


@dataclass
class TaskResult:
    coord: tuple[int, int]
    embedding_id: List[str]
    local_file_path: List[str]
    score: List[float]

class AsyncTaskHandler:
    """Handler for managing asynchronous search tasks with embeddings"""
    
    def __init__(
        self, 
        depleting_collection_name: str, 
        read_only_collection_name: str, 
        real_client: qdrant_client.AsyncQdrantClient, 
        fake_client: FakeQdrantClient,
        debug_delay: float = 0.0  # Add debug delay parameter
    ):
        self.real_client = real_client
        self.fake_client = fake_client
        self.depleting_collection_name = depleting_collection_name
        self.read_only_collection_name = read_only_collection_name
        self.tasks = []
        self.results_by_id = {}
        self.debug_delay = debug_delay  # Store the debug delay
    
    async def search_with_embedding(
        self, 
        results_limit: int,
        coord: str, 
        neighbour_ids: List[str],
    ) -> TaskResult | Exception:
        """Perform a search with the given embedding and return results"""
        try:
            # Add optional debugging delay to see sequential operations clearly
            if self.debug_delay > 0:
                # print(f"DEBUG: Adding {self.debug_delay}s delay before processing coordinate {coord}")
                await asyncio.sleep(self.debug_delay)
                
            if len(neighbour_ids) == 0:
                # print("No neighbour ids found")
                try:
                    # get a random point - we want this depleted so its removed from collection
                    points = await async_get_random_point(client=self.real_client, collection_name=self.depleting_collection_name)
                    id = [points[0].id]
                    filepath = [points[0].payload["filename"]]
                    score = [points[0].score]
                except ValueError as e:
                    print(f"Error getting random item: {e}")
                    return e
            elif len(neighbour_ids) > 0:
                # get neighbour ids - we want from write only as need reference
                embedding_average = await async_get_embedding_average(self.real_client, neighbour_ids, self.read_only_collection_name)
                # get depleting match so its removed from collection
                res = await async_get_closest_match(
                    client=self.real_client,
                    collection_name=self.depleting_collection_name,
                    vector=embedding_average,
                    limit=results_limit,
                    with_payload=True,
                    with_vectors=False
                    )

                if len(res) == 0:
                    raise ValueError("No closest match found: collection probably depleted")
                id = [res_.id for res_ in res]#res[0].id
                filepath = [res_.payload["filename"] for res_ in res]#res[0].payload["filename"]
                score = [res_.score for res_ in res]#res[0].score

            # results = await async_get_random_point(self.real_client, self.collection_name)
            # print(points[0].vector[0])
            return TaskResult(
                coord=coord,
                embedding_id=id,
                local_file_path=filepath,
                score=score
            )
        except Exception as e:
            return e
    
    async def process_embeddings(
        self, 
        results_limit: int,
        neighbour_ids: Dict[tuple[int,int], List[str]], 
        force_sequential: bool = False,
        delete_after_processing: bool = True
    ) -> List[Any]:
        """Process embeddings either in parallel or sequentially.
        
        Args:
            neighbour_ids: Dictionary mapping coordinates to lists of neighbour IDs
            force_sequential: If True, process embeddings one by one (for debugging)
            delete_after_processing: Whether to delete points after processing
            
        Returns:
            List of results (either TaskResult objects or exceptions)
        """
        # Clear previous tasks
        self.tasks = []

        if force_sequential:
            # print(f"Processing {len(neighbour_ids)} embeddings SEQUENTIALLY")
            # Process one by one in sequence
            all_results = []
            for coord, n_ids in neighbour_ids.items():
                # print(f"Processing coordinate {coord} sequentially...")
                # Process directly (await each operation individually)
                try:
                    result = await self.search_with_embedding(results_limit, coord, n_ids)
                    all_results.append(result)
                    
                    # Delete immediately in sequential mode if requested
                    if delete_after_processing and isinstance(result, TaskResult):
                        # print(f"Immediately deleting point {result.embedding_id} after processing")
                        await self.real_client.delete(
                            collection_name=self.depleting_collection_name,
                            points_selector=models.PointIdsList(
                                points=[result.embedding_id[0]]
                            ),
                            wait=True
                        )
                    
                    # print(f"Completed processing coordinate {coord}")
                except Exception as e:
                    print(f"Error processing coordinate {coord}: {e}")
                    all_results.append(e)
            return all_results
        else:
            # print(f"Processing {len(neighbour_ids)} embeddings in PARALLEL")
            # each worker gets a coordinate, and a list of neighbours ids
            # Process concurrently using tasks
            for coord, n_ids in neighbour_ids.items():
                task = asyncio.create_task(
                    self.search_with_embedding(results_limit, coord, n_ids)
                )
                self.tasks.append(task)
            
            if not self.tasks:
                print("No tasks to run")
                return []
            
            # Wait for all tasks to complete and gather results
            all_results = await asyncio.gather(*self.tasks, return_exceptions=True)
            
            return all_results


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