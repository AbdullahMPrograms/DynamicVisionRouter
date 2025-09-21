#!/usr/bin/env python3
"""
Test script to see what usage statistics llama.cpp actually outputs.
Fill in your llama.cpp endpoint details below and run this script.
"""

import json
import asyncio
import aiohttp

# ============================================================================
# CONFIGURATION - Fill these in with your llama.cpp server details
# ============================================================================
OPENAI_API_URL = "http://192.168.2.208:8081/v1/chat/completions"  # Replace with your endpoint
OPENAI_API_KEY = "none"  # Replace with your API key (or "none" if not needed)
MODEL_NAME = "GPT-OSS-120B"  # Replace with your model name

# Test message
TEST_MESSAGE = "Write a short poem about coding."

async def test_streaming_usage():
    """Test streaming with usage statistics"""
    print("=== TESTING STREAMING WITH USAGE ===")
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": TEST_MESSAGE}
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    print("\n--- Streaming Response ---")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(OPENAI_API_URL, headers=headers, json=payload) as response:
            if response.status != 200:
                print(f"Error: HTTP {response.status}")
                print(await response.text())
                return
            
            content_chunks = []
            usage_data = None
            
            async for line in response.content:
                if line and line.startswith(b"data: "):
                    line_text = line[6:].decode("utf-8").strip()
                    
                    if line_text == "[DONE]":
                        print("\n[DONE] received")
                        break
                    
                    try:
                        data = json.loads(line_text)
                        print(f"Raw frame: {json.dumps(data)}")
                        
                        # Check for usage information
                        if "usage" in data:
                            usage_data = data["usage"]
                            print(f"\n🎯 USAGE FOUND: {json.dumps(usage_data, indent=2)}")
                        
                        # Collect content
                        if ("choices" in data and 
                            len(data["choices"]) > 0 and 
                            "delta" in data["choices"][0] and
                            "content" in data["choices"][0]["delta"]):
                            content = data["choices"][0]["delta"]["content"]
                            if content:
                                content_chunks.append(content)
                                print(f"Content chunk: {repr(content)}")
                    
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        print(f"Raw line: {repr(line_text)}")
            
            print(f"\n--- Final Results ---")
            print(f"Complete response: {''.join(content_chunks)}")
            if usage_data:
                print(f"Final usage data: {json.dumps(usage_data, indent=2)}")
                print("\nUsage fields detected:")
                for key, value in usage_data.items():
                    print(f"  {key}: {value} ({type(value).__name__})")
            else:
                print("❌ No usage data found in streaming response")

async def test_non_streaming_usage():
    """Test non-streaming with usage statistics"""
    print("\n\n=== TESTING NON-STREAMING WITH USAGE ===")
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": TEST_MESSAGE}
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(OPENAI_API_URL, headers=headers, json=payload) as response:
            if response.status != 200:
                print(f"Error: HTTP {response.status}")
                print(await response.text())
                return
            
            result = await response.json()
            print(f"\n--- Full Response ---")
            print(f"Raw response: {json.dumps(result, indent=2)}")
            
            if "usage" in result:
                usage_data = result["usage"]
                print(f"\n🎯 USAGE FOUND: {json.dumps(usage_data, indent=2)}")
                print("\nUsage fields detected:")
                for key, value in usage_data.items():
                    print(f"  {key}: {value} ({type(value).__name__})")
            else:
                print("❌ No usage data found in non-streaming response")
            
            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                print(f"\nResponse content: {content}")

async def main():
    """Run both tests"""
    print("Llama.cpp Usage Statistics Test")
    print("=" * 50)
    
    # Validate configuration
    if OPENAI_API_URL == "http://localhost:8080/v1/chat/completions":
        print("⚠️  Please update OPENAI_API_URL with your actual endpoint")
    if OPENAI_API_KEY == "your-api-key-here":
        print("⚠️  Please update OPENAI_API_KEY with your actual key (or 'none')")
    if MODEL_NAME == "your-model-name":
        print("⚠️  Please update MODEL_NAME with your actual model name")
    
    print(f"\nTesting endpoint: {OPENAI_API_URL}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Test message: {TEST_MESSAGE}")
    
    try:
        # Test streaming first
        await test_streaming_usage()
        
        # Then test non-streaming
        await test_non_streaming_usage()
        
        print("\n" + "=" * 50)
        print("Test completed! Check the usage data above to see what fields")
        print("your llama.cpp server actually provides.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())