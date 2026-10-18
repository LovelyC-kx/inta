"""
Intan RHX Demo Mode Validation Script.

Run this script while Intan RHX is running in Demo mode with TCP enabled.

Prerequisites:
    1. Start Intan RHX -> Demo Mode -> RHS Controller -> 30 kS/s
    2. Menu: Network -> Remote TCP Control
    3. Set Command port to 5000, Waveform Output to 5001, Spike Output to 5002
    4. Click Connect on each (status should show "Pending")
    5. Run this script

This script validates:
    - TCP connection establishment
    - Basic Get/Set commands
    - Sample rate query
    - Run/Stop control
"""

import sys
import time

# Allow running from project root
sys.path.insert(0, "..")
from tcp_client import IntanTCPClient


def validate_tcp_connection(host="127.0.0.1", command_port=5000):
    """Run validation steps for Intan TCP interface."""

    print("=" * 60)
    print("Intan RHX TCP Validation")
    print("=" * 60)
    print(f"Connecting to {host}:{command_port}...")

    client = IntanTCPClient(host=host, command_port=command_port)

    try:
        # Step 1: Connect
        client.connect_command(timeout=5.0)
        print("[OK] Command socket connected")

        # Step 2: Query run mode
        response = client.get("RunMode")
        print(f"[OK] Get RunMode -> {response}")

        # Step 3: Query sample rate
        rate = client.get_sample_rate()
        print(f"[OK] Sample rate: {rate} Hz")

        # Step 4: Start acquisition
        response = client.run()
        print(f"[OK] Set RunMode Run -> {response}")
        time.sleep(1)

        # Step 5: Stop acquisition
        response = client.stop()
        print(f"[OK] Set RunMode Stop -> {response}")

        # Step 6: Query an invalid parameter (error handling)
        try:
            response = client.get("NonexistentParameter")
            print(f"[INFO] Invalid param response: {response}")
        except Exception as e:
            print(f"[INFO] Invalid param error (expected): {e}")

        print("\n" + "=" * 60)
        print("TCP validation complete. All basic commands work.")
        print("=" * 60)

    except ConnectionRefusedError:
        print("[ERROR] Connection refused. Is Intan RHX running with TCP enabled?")
        print("  1. Start Intan RHX in Demo mode (RHS Controller)")
        print("  2. Network -> Remote TCP Control")
        print("  3. Set port to 5000 and click Connect")
    except TimeoutError as e:
        print(f"[ERROR] Timeout: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
    finally:
        client.disconnect()
        print("Connection closed.")


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    validate_tcp_connection(host=host, command_port=port)
