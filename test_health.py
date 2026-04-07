"""
Quick test to see if the server can be started without Milvus
"""
import threading
import time
import requests
from test_server import main

def start_server():
    try:
        main()  # This will block, so we run it in a thread
    except Exception as e:
        print(f"Server error: {e}")

# Start server in background thread
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# Wait a bit for server to start
time.sleep(3)

try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print("Server is running!")
    print(f"Health check response: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.exceptions.ConnectionError:
    print("Server is not accessible at http://localhost:8000/health")
    print("This could be because:")
    print("1. Server is still starting up")
    print("2. Server failed to start due to other issues")
    print("3. Server is running on a different port")
except Exception as e:
    print(f"Error during health check: {e}")