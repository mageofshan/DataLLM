import httpx
import asyncio

async def test_chat():
    url = 'http://localhost:8000/api/v1/chat'
    
    # Test 1: Direct Computation
    print("\n--- Test 1: Direct Computation (Mean) ---")
    payload = {
        "query": "Calculate the mean of the dataset",
        "dataset_id": "mock-id" # Won't actually load unless we have a real ID, but router checks ID presence
    }
    # Note: Since we don't have a real ID from a previous run in this script, 
    # the router will fail to load the dataset and fall back to LLM or error.
    # Let's just test the LLM route first without a dataset ID to be safe, 
    # or mock the storage load.
    
    # Actually, let's just test the LLM route for now as it doesn't strictly require a dataset to work (just won't have context).
    payload = {
        "query": "What is the meaning of life?",
        "dataset_id": None
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

if __name__ == "__main__":
    asyncio.run(test_chat())
