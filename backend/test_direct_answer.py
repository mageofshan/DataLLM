import httpx
import asyncio

async def test_direct_answer():
    base_url = "http://localhost:8000/api/v1"
    
    # Upload the test file
    print("1. Uploading test stock data...")
    with open("/Users/saishantanusivakumaran/DataLLM/test_stock_data.csv", "rb") as f:
        files = {'file': ('test_stock_data.csv', f, 'text/csv')}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{base_url}/upload", files=files)
            
            if response.status_code != 200:
                print(f"Upload failed: {response.text}")
                return
                
            data = response.json()
            dataset_id = data['dataset_id']
            print(f"✓ Upload success! Dataset ID: {dataset_id}")
            
            # Test direct answer (should NOT include python_code)
            print("\n2. Testing direct answer query...")
            chat_payload = {
                "query": "what is the average open price?",
                "dataset_id": dataset_id
            }
            response = await client.post(f"{base_url}/chat", json=chat_payload)
            result = response.json()
            
            print(f"\n✓ Response received:")
            print(f"  Summary: {result['response']}")
            print(f"  Insights: {result['data']['insights']}")
            print(f"  Python Code: {result['data']['python_code']}")
            print(f"  Python Params: {result['data']['python_params']}")
            
            if result['data']['python_code'] is None:
                print("\n✅ SUCCESS: No unnecessary code was returned!")
            else:
                print("\n⚠️  WARNING: Code was still returned when it shouldn't be")

if __name__ == "__main__":
    asyncio.run(test_direct_answer())
