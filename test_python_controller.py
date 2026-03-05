#!/usr/bin/env python3
"""
Test script for the Python gesture controller
Run this to verify everything is working correctly
"""

import requests
import time
import json

SERVER_URL = "http://localhost:5001"

def test_server_connection():
    """Test if the gesture server is running"""
    print("1. Testing server connection...")
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Server running on port 5001")
            print(f"   ✓ Running: {data.get('running', 'Unknown')}")
            print(f"   ✓ Detection method: {data.get('detection_method', 'Unknown')}")
            print(f"   ✓ Tracking: {data.get('tracking', 'Unknown')}")
            return True
        else:
            print(f"   ✗ Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Cannot connect to server: {e}")
        print("   Make sure dobot_gesture_control.py is running!")
        return False

def test_gesture_detection():
    """Test gesture detection"""
    print("\n2. Testing gesture detection...")
    try:
        response = requests.get(f"{SERVER_URL}/detection", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('hand_detected'):
                print(f"   ✓ Hand detected!")
                print(f"   ✓ Gesture: {data['gesture_id']} ({data['gesture_name']})")
                print(f"   ✓ Confidence: {data['confidence']:.0%}")
                print(f"   ✓ Method: {data['method']}")
                return data['gesture_id']
            else:
                print(f"   ⚠ No hand detected")
                print(f"   ⚠ Status: {data.get('gesture_name', 'Unknown')}")
                print("   Show your hand to the camera!")
                return None
        else:
            print(f"   ✗ Detection endpoint returned {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Detection test failed: {e}")
        return None

def test_robot_status():
    """Test robot connection status"""
    print("\n3. Testing robot status...")
    try:
        response = requests.get(f"{SERVER_URL}/detection", timeout=5)
        if response.status_code == 200:
            data = response.json()
            robot_info = data.get('robot', {})
            if robot_info:
                print(f"   ✓ Robot connected: {robot_info.get('connected', False)}")
                print(f"   ✓ Robot enabled: {robot_info.get('enabled', False)}")
                print(f"   ✓ Robot IP: {robot_info.get('ip', 'Unknown')}")
                print(f"   ✓ Current movement: {robot_info.get('current_movement', 'None')}")
                if robot_info.get('current_pose'):
                    pose = robot_info['current_pose']
                    print(f"   ✓ Position: X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f}")
                return robot_info.get('connected', False) and robot_info.get('enabled', False)
            else:
                print("   ⚠ No robot information available")
                return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Robot status test failed: {e}")
        return False

def test_tracking_control():
    """Test tracking start/stop"""
    print("\n4. Testing tracking control...")
    try:
        # Test stop
        response = requests.post(f"{SERVER_URL}/tracking/stop", timeout=5)
        if response.status_code == 200:
            print("   ✓ Tracking stop works")
        
        # Test start
        response = requests.post(f"{SERVER_URL}/tracking/start", timeout=5)
        if response.status_code == 200:
            print("   ✓ Tracking start works")
            return True
        else:
            print(f"   ✗ Tracking control failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Tracking control test failed: {e}")
        return False

def test_gesture_movement():
    """Test if gestures trigger movement commands"""
    print("\n5. Testing gesture movement...")
    print("   Show different gestures for 10 seconds...")
    
    movements_detected = []
    for i in range(20):  # 10 seconds at 0.5s intervals
        try:
            response = requests.get(f"{SERVER_URL}/detection", timeout=2)
            if response.status_code == 200:
                data = response.json()
                if data.get('hand_detected'):
                    gesture_id = data['gesture_id']
                    movement = data.get('robot_movement')
                    print(f"   Gesture {gesture_id}: {data['gesture_name']} → {movement or 'STOPPED'}")
                    if movement and movement not in movements_detected:
                        movements_detected.append(movement)
                else:
                    print("   No hand detected")
        except:
            print("   Detection failed")
        
        time.sleep(0.5)
    
    if movements_detected:
        print(f"   ✓ Detected movements: {', '.join(movements_detected)}")
        return True
    else:
        print("   ⚠ No movements detected - try showing clearer gestures")
        return False

def main():
    print("=" * 60)
    print("  Python Gesture Controller Test")
    print("=" * 60)
    
    # Test 1: Server connection
    if not test_server_connection():
        print("\n❌ FAILED: Server not running")
        print("Start the controller with: python dobot_gesture_control.py")
        return
    
    # Test 2: Gesture detection
    gesture_detected = test_gesture_detection()
    
    # Test 3: Robot status
    robot_ready = test_robot_status()
    
    # Test 4: Tracking control
    tracking_works = test_tracking_control()
    
    # Test 5: Movement detection
    movement_works = test_gesture_movement()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    print(f"Server Connection:    {'✓ PASS' if True else '✗ FAIL'}")
    print(f"Gesture Detection:    {'✓ PASS' if gesture_detected else '⚠ NO HAND'}")
    print(f"Robot Connection:     {'✓ PASS' if robot_ready else '✗ FAIL'}")
    print(f"Tracking Control:     {'✓ PASS' if tracking_works else '✗ FAIL'}")
    print(f"Movement Commands:    {'✓ PASS' if movement_works else '⚠ NO MOVEMENTS'}")
    
    if robot_ready and movement_works:
        print("\n🎉 SUCCESS: Everything is working!")
        print("Your hand gestures should now control the robot.")
        print("Open http://localhost:5001 to see the web dashboard.")
    elif not robot_ready:
        print("\n⚠ ISSUE: Robot not connected or enabled")
        print("Make sure the robot is connected via TCP/IP and enabled.")
    elif not movement_works:
        print("\n⚠ ISSUE: Gestures detected but no movements")
        print("Try showing clearer hand gestures to the camera.")
    
    print("\nGesture mapping:")
    print("  1 (Index) → UP")
    print("  2 (Index+Middle) → FORWARD") 
    print("  3 (Index+Middle+Ring) → RIGHT")
    print("  4 (4 fingers) → DOWN")
    print("  5 (All fingers) → STOP")
    print("  6 (Thumb) → LEFT")
    print("  7 (Thumb+Index) → BACKWARD")
    print("  8 (Thumb+Index+Middle) → ROTATE CW")
    print("  9 (4 fingers+Thumb) → ROTATE CCW")
    print("  10 (Fist) → STOP")

if __name__ == "__main__":
    main()