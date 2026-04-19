#!/usr/bin/env python3
import socket
import os
import sys
import time

# =====================
# CONFIGURATION
# =====================
sock_address = os.environ.get("SO3LR_SOCKET", "/scratch/v15/hp2306/so3lr_socket")
RESULT_TIMEOUT = 300.0   # Max seconds to wait for .result file to appear
POLL_INTERVAL  = 0.01   # Seconds between polls

# =====================
# ARGUMENT CHECK
# =====================
if len(sys.argv) < 2:
    print("Usage: client_SO3LR.py <input_file>")
    sys.exit(1)

input_file  = sys.argv[1]
result_file = input_file + '.result'

# =====================
# CLEAN UP OLD RESULT
# =====================
if os.path.exists(result_file):
    os.remove(result_file)

# =====================
# SEND REQUEST
# =====================
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.settimeout(300.0)

try:
    client.connect(sock_address)
    client.send(input_file.encode('utf-8'))

    response = client.recv(2).decode('utf-8')

    if response.startswith('E'):
        print(f"ERROR: Server reported a failure processing {input_file}")
        sys.exit(1)

except socket.timeout:
    print("ERROR: Timed out waiting for server response.")
    sys.exit(1)

except socket.error as e:
    print(f"ERROR: Socket error: {e}")
    sys.exit(1)

finally:
    client.close()

# =====================
# WAIT FOR RESULT FILE
# =====================
elapsed = 0.0
while not os.path.exists(result_file) and elapsed < RESULT_TIMEOUT:
    time.sleep(POLL_INTERVAL)
    elapsed += POLL_INTERVAL

if not os.path.exists(result_file):
    print(f"ERROR: Result file never appeared: {result_file}")
    sys.exit(1)

print(f"Result ready: {result_file}")
