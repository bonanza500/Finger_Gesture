#!/usr/bin/env python3
"""
Debug script for robot communication
Run this to test robot commands directly
"""

import socket
import time
import re

ROBOT_IP = "192.168.1.6"
DASHBOARD_PORT = 29999
MOVE_PORT = 30003

def test_robot_connection():
    """Test basic robot connection and commands"""
    print("=" * 50)
    print("  Robot Communication Debug")
    print("=" * 50)
    
    # Test dashboard connection
    print(f"\n1. Testing dashboard connection to {ROBOT_IP}:{DASHBOARD_PORT}")
    try:
        dashboard = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dashboard.settimeout(5)
        dashboard.connect((ROBOT_IP, DASHBOARD_PORT))
        print("   ✓ Dashboard connected")
        
        # Test basic commands
        def send_dashboard(cmd):
            dashboard.sendall((cmd + "\n").encode("utf-8"))
            resp = dashboard.recv(1024).decode("utf-8").strip()
            print(f"   CMD: {cmd}")
            print(f"   RESP: {resp}")
            return resp
        
        # Get robot status
        print("\n2. Testing basic dashboard commands:")
        send_dashboard("RequestControl()")
        time.sleep(0.5)
        send_dashboard("ClearError()")
        time.sleep(0.5)
        
        # Get current pose
        print("\n3. Testing GetPose():")
        pose_resp = send_dashboard("GetPose()")
        
        # Parse pose
        if pose_resp:
            numbers = re.findall(r'[-\d.]+', pose_resp)
            if len(numbers) >= 6:
                pose = [float(n) for n in numbers[-6:]]
                print(f"   Parsed pose: X={pose[0]:.2f}, Y={pose[1]:.2f}, Z={pose[2]:.2f}")
                print(f"                RX={pose[3]:.2f}, RY={pose[4]:.2f}, RZ={pose[5]:.2f}")
            else:
                print(f"   Could not parse pose from: {pose_resp}")
        
        dashboard.close()
        
    except Exception as e:
        print(f"   ✗ Dashboard connection failed: {e}")
        return False
    
    # Test move connection
    print(f"\n4. Testing move connection to {ROBOT_IP}:{MOVE_PORT}")
    try:
        mover = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mover.settimeout(5)
        mover.connect((ROBOT_IP, MOVE_PORT))
        print("   ✓ Move port connected")
        
        def send_move(cmd):
            mover.sendall((cmd + "\n").encode("utf-8"))
            resp = mover.recv(1024).decode("utf-8").strip()
            print(f"   CMD: {cmd}")
            print(f"   RESP: {resp}")
            return resp
        
        # Test a small movement (CAREFUL!)
        print("\n5. Testing small movement (UP 10mm):")
        print("   WARNING: Robot will move in 3 seconds!")
        time.sleep(1)
        print("   3...")
        time.sleep(1)
        print("   2...")
        time.sleep(1)
        print("   1...")
        time.sleep(1)
        
        # Get current pose again for movement
        dashboard2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dashboard2.settimeout(5)
        dashboard2.connect((ROBOT_IP, DASHBOARD_PORT))
        dashboard2.sendall("GetPose()\n".encode("utf-8"))
        pose_resp = dashboard2.recv(1024).decode("utf-8").strip()
        dashboard2.close()
        
        numbers = re.findall(r'[-\d.]+', pose_resp)
        if len(numbers) >= 6:
            current_pose = [float(n) for n in numbers[-6:]]
            new_pose = current_pose.copy()
            new_pose[2] += 10  # Move Z up by 10mm
            
            pose_str = ",".join([f"{p:.2f}" for p in new_pose])
            move_cmd = f"MovL({pose_str})"
            send_move(move_cmd)
            
            print("   Movement command sent!")
            time.sleep(2)
            
            # Move back
            pose_str = ",".join([f"{p:.2f}" for p in current_pose])
            move_cmd = f"MovL({pose_str})"
            send_move(move_cmd)
            print("   Moved back to original position")
            
        else:
            print("   Could not get current pose for movement test")
        
        mover.close()
        
    except Exception as e:
        print(f"   ✗ Move connection failed: {e}")
        return False
    
    print("\n✓ Robot communication test complete!")
    return True

def test_pose_parsing():
    """Test different pose response formats"""
    print("\n" + "=" * 50)
    print("  Pose Parsing Test")
    print("=" * 50)
    
    # Common response formats from Dobot
    test_responses = [
        "{0,0,[100.00,200.00,300.00,0.00,0.00,0.00]}",
        "{0,[100.00,200.00,300.00,0.00,0.00,0.00]}",
        "[100.00,200.00,300.00,0.00,0.00,0.00]",
        "100.00,200.00,300.00,0.00,0.00,0.00",
    ]
    
    for i, resp in enumerate(test_responses, 1):
        print(f"\n{i}. Testing format: {resp}")
        
        # Method 1: Look for array
        match = re.search(r'\[([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\]', resp)
        if match:
            pose = [float(match.group(i)) for i in range(1, 7)]
            print(f"   Method 1 (array): {pose}")
        else:
            print("   Method 1 (array): Failed")
        
        # Method 2: Find all numbers
        numbers = re.findall(r'[-\d.]+', resp)
        if len(numbers) >= 6:
            pose = [float(n) for n in numbers[-6:]]
            print(f"   Method 2 (numbers): {pose}")
        else:
            print(f"   Method 2 (numbers): Failed - only {len(numbers)} numbers")

if __name__ == "__main__":
    print("Robot Debug Script")
    print("This will test robot communication and movement")
    print("\nMake sure:")
    print("1. Robot is connected and enabled")
    print("2. Robot is in a safe position")
    print("3. Workspace is clear")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    test_pose_parsing()
    test_robot_connection()