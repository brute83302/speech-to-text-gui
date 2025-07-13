#!/usr/bin/env python3

# Test script to identify the issue
try:
    print("Testing imports...")
    import tkinter as tk
    print("✓ tkinter imported")
    
    from tkinter import ttk
    print("✓ ttk imported")
    
    import sounddevice as sd
    print("✓ sounddevice imported")
    
    import soundfile as sf
    print("✓ soundfile imported")
    
    import numpy as np
    print("✓ numpy imported")
    
    import whisper
    print("✓ whisper imported")
    
    print("Creating basic window...")
    root = tk.Tk()
    root.title("Test")
    root.geometry("300x200")
    
    label = tk.Label(root, text="Test GUI")
    label.pack(pady=50)
    
    print("✓ Basic GUI created successfully")
    print("Starting mainloop (close window to continue)...")
    
    root.mainloop()
    print("✓ GUI closed normally")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()