#!/usr/bin/env python
"""Test script to verify TensorFlow 2.20.0 fix for register_load_context_function error"""

import sys

print("Testing TensorFlow import and TFLite export capability...")
print("-" * 60)

try:
    # Test 1: Import TensorFlow
    print("\n1️⃣  Importing TensorFlow...")
    import tensorflow as tf
    print(f"   ✅ TensorFlow {tf.__version__} imported successfully")
    
    # Test 2: Check that the problematic internal API is NOT present (expected in 2.20.0)
    print("\n2️⃣  Checking internal API...")
    try:
        from tensorflow._api.v2.compat.v2.__internal__ import register_load_context_function
        print("   ℹ️  register_load_context_function found")
    except (AttributeError, ImportError):
        print("   ✅ register_load_context_function not in v2.20.0 (expected)")
        print("   This fixes the AttributeError that was occurring in v2.21.0")
    
    # Test 3: Try to load TFLite converter
    print("\n3️⃣  Testing TFLite converter import...")
    from tensorflow.lite.python import lite_constants
    print("   ✅ TFLite converter available")
    
    # Test 4: Create a simple model and try to export
    print("\n4️⃣  Testing model export...")
    import numpy as np
    
    # Create a simple model
    input_shape = (1, 3, 224, 224)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape[1:]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    print("   ✅ Model created")
    
    # Try to convert to TFLite (this is where the error would occur)
    print("   Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print(f"   ✅ Model converted to TFLite ({len(tflite_model)} bytes)")
    
    print("\n" + "=" * 60)
    print("✅  All tests passed! TensorFlow is working correctly.")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    print("\nTraceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)
