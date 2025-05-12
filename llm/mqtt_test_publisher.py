# llm/mqtt_test_publisher.py

import os
import json
import time
import sys
import paho.mqtt.client as paho
from paho import mqtt

# Get the parent directory of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path to find the configs module
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Now import settings from the configs module
    from configs import settings
except ImportError as e:
    print(f"Error: Could not import settings from configs.py. Ensure it exists and is accessible.")
    print(f"Current sys.path: {sys.path}")
    print(f"ImportError: {e}")
    sys.exit(1)

connected = False

def on_connect(client, userdata, flags, rc, properties=None):
    global connected
    if rc == 0:
        print("Connection successful! Result code: {rc}".format(rc=str(rc)))
        connected = True
    else:
        print("Connection failed! Result code: {rc}".format(rc=str(rc)))
        connected = False

def on_publish(client, userdata, mid, properties=None):
    print("Message Published with MID: "+str(mid))

def on_disconnect(client, userdata, rc, properties=None):
    global connected
    print("Client Disconnected. Result code: " + str(rc))
    connected = False

# --- Configuration --- 
client_id = "llm_test_publisher_" + str(int(time.time())) # Unique client ID

# Use settings from configs.py
MQTT_BROKER_HOST = settings.MQTT_BROKER_HOST
MQTT_BROKER_PORT = settings.MQTT_BROKER_PORT
MQTT_USERNAME = settings.MQTT_USERNAME
MQTT_PASSWORD = settings.MQTT_PASSWORD
MQTT_PROCESS_TOPIC = settings.MQTT_PROCESS_TOPIC

# --- Payload to send --- 
# This should be a JSON structure that process_llm_request_event_data expects
# Modify this payload as needed for your actual test case
test_payload = {
    "request_id": f"test-{int(time.time())}",
    "infraction_code": "code-123",
    "event_type": "weapons",
    "app_type": "lucam",
    "image_url": "/Users/stanleysalvatierra/Desktop/2024/lucam/face/data/1.png",
    "details": {
        "param1": "value1",
        "param2": 12345
    },
    "timestamp": time.time()
}

# --- MQTT Client Setup --- 
client = paho.Client(client_id=client_id)
client.on_connect = on_connect
client.on_publish = on_publish
client.on_disconnect = on_disconnect

# Enable logging for debugging
# client.enable_logging(logger=None)

# Set username and password
if MQTT_USERNAME and MQTT_PASSWORD:
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# Set TLS options if needed (uncomment and configure if your broker uses TLS)
# client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)

# --- Connect and Publish --- 
print(f"Attempting to connect to broker: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
except Exception as e:
    print(f"Error connecting to MQTT broker: {e}")
    sys.exit(1)

# Start the network loop in a non-blocking way
client.loop_start()

# Wait for connection
connection_timeout = 10 # seconds
start_time = time.time()
while not connected and time.time() - start_time < connection_timeout:
    print("Waiting for connection...")
    time.sleep(1)

if not connected:
    print("Connection timed out. Exiting.")
    client.loop_stop()
    sys.exit(1)

# Publish the message
payload_str = json.dumps(test_payload)
print(f"Publishing to topic '{MQTT_PROCESS_TOPIC}': {payload_str}")
msg_info = client.publish(MQTT_PROCESS_TOPIC, payload=payload_str, qos=1)

# Wait for the message to be published
try:
    msg_info.wait_for_publish(timeout=5)
    if msg_info.is_published():
        print("Publish confirmed.")
    else:
        print("Publish confirmation timed out.")
except ValueError:
    print("Publish failed immediately (e.g., client disconnected).")
except RuntimeError as e:
    print(f"Runtime error during publish wait: {e}")

# --- Disconnect --- 
print("Disconnecting...")
time.sleep(1) # Give a moment before disconnecting
client.loop_stop()
client.disconnect()

print("Script finished.") 