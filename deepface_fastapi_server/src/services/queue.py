import paho.mqtt.client as paho
from paho import mqtt
import sys
import time
import json
import threading

try:
    # Now import settings from the configs module
    from config import settings
except ImportError as e:
    print(f"Error: Could not import settings from configs.py. Ensure it exists and is accessible.")
    print(f"Current sys.path: {sys.path}")
    print(f"ImportError: {e}")
    sys.exit(1)

class MQTTClientSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            
            # Configuration
            self.client_id = f"mqtt_singleton_client_{int(time.time())}"
            self.host = settings.MQTT_BROKER_HOST
            self.port = settings.MQTT_BROKER_PORT

            # State
            self.connected = False
            self._connection_established = threading.Event() # Event to signal connection

            # Client Setup
            self.client = paho.Client(client_id=self.client_id, protocol=paho.MQTTv311)
            self.client.on_connect = self._on_connect
            self.client.on_publish = self._on_publish
            self.client.on_disconnect = self._on_disconnect


            # TLS (Optional - configure as needed)
            # self.client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
            
            self._initialized = True


    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"MQTT Singleton: Connection successful to {self.host}:{self.port}")
            self.connected = True
            self._connection_established.set() # Signal that connection is ready
        else:
            print(f"MQTT Singleton: Connection failed! Result code: {rc}")
            self.connected = False
            self._connection_established.clear()

    def _on_publish(self, client, userdata, mid, properties=None):
        print(f"MQTT Singleton: Message Published with MID: {mid}")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        print(f"MQTT Singleton: Client Disconnected. Result code: {rc}")
        self.connected = False
        self._connection_established.clear()
        # Optional: Implement automatic reconnection logic here if needed

    def connect(self):
        if not self.connected:
            print(f"MQTT Singleton: Attempting to connect to broker: {self.host}:{self.port}")
            try:
                self.client.connect(self.host, self.port, keepalive=60)
                self.client.loop_start() # Start network loop
                print("MQTT Singleton: Waiting for connection establishment...")
                # Wait for the connection to be established, with a timeout
                if not self._connection_established.wait(timeout=10):
                     print("MQTT Singleton: Connection attempt timed out.")
                     self.disconnect() # Clean up if connection failed
                     raise ConnectionError("MQTT Connection timed out")
                print("MQTT Singleton: Connection established and loop started.")

            except Exception as e:
                print(f"MQTT Singleton: Error connecting to MQTT broker: {e}")
                self.disconnect() # Ensure cleanup on error
                raise ConnectionError(f"MQTT Connection failed: {e}") from e


    def publish(self, topic, payload, qos=1):
        if not self.connected or not self._connection_established.is_set():
             print("MQTT Singleton: Not connected. Attempting to connect first.")
             try:
                self.connect()
             except ConnectionError as e:
                 print(f"MQTT Singleton: Failed to connect before publishing: {e}")
                 return False # Indicate failure


        if isinstance(payload, dict):
            payload_str = json.dumps(payload)
        elif isinstance(payload, str):
             payload_str = payload
        else:
            print("MQTT Singleton: Payload must be a dict or a JSON string.")
            return False # Indicate failure


        print(f"MQTT Singleton: Publishing to topic '{topic}': {payload_str[:100]}...") # Log truncated payload
        msg_info = self.client.publish(topic, payload=payload_str, qos=qos)

        try:
            # Wait for publish confirmation with a timeout
            msg_info.wait_for_publish(timeout=5)
            if msg_info.is_published():
                print(f"MQTT Singleton: Publish to {topic} confirmed (MID: {msg_info.mid}).")
                return True # Indicate success
            else:
                print(f"MQTT Singleton: Publish to {topic} confirmation timed out (MID: {msg_info.mid}).")
                return False # Indicate failure (timeout)
        except ValueError:
            print(f"MQTT Singleton: Publish to {topic} failed immediately (e.g., client disconnected).")
            return False # Indicate failure
        except RuntimeError as e:
            print(f"MQTT Singleton: Runtime error during publish wait for topic {topic}: {e}")
            return False # Indicate failure


    def disconnect(self):
        if self.client and self.client.is_connected():
             print("MQTT Singleton: Disconnecting...")
             self.client.loop_stop() # Stop the network loop gracefully
             self.client.disconnect()
             print("MQTT Singleton: Disconnected.")
        else:
             print("MQTT Singleton: Already disconnected or loop not running.")
        self.connected = False
        self._connection_established.clear()



mqtt_client = None
def get_mqtt_client():
    global mqtt_client
    if mqtt_client is None:
        mqtt_client = MQTTClientSingleton()
    return mqtt_client


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    print("Starting MQTT Singleton example...")
    mqtt_client = get_mqtt_client()
    try:
        mqtt_client.connect() # Ensure connection is established

        # Example payload
        test_payload = {
            "request_id": f"test-singleton-{int(time.time())}",
            "data": "some test data"
        }

        # Use the process topic from settings
        topic_to_publish = settings.MQTT_LLM_TOPIC

        # Publish the message
        success = mqtt_client.publish(topic_to_publish, test_payload)

        if success:
            print("Example: Message published successfully.")
        else:
            print("Example: Message publication failed.")

        time.sleep(2) # Keep running briefly to allow callbacks

    except ConnectionError as e:
        print(f"Example: Could not connect to MQTT: {e}")
    except Exception as e:
        print(f"Example: An unexpected error occurred: {e}")
    finally:
        # Ensure disconnection
        print("Example: Cleaning up...")
        mqtt_client.disconnect()

    print("Example finished.") 