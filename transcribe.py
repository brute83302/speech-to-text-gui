#!/usr/bin/env python3

import whisper
import sys
import os
from pathlib import Path

def transcribe_audio(audio_file, model_size="turbo"):
    """
    Transcribe audio file using Whisper
    
    Args:
        audio_file (str): Path to audio file
        model_size (str): Whisper model size (tiny, base, small, medium, large, turbo)
    
    Returns:
        str: Transcribed text
    """
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing: {audio_file}")
    result = model.transcribe(audio_file)
    
    return result["text"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file> [model_size]")
        print("Model sizes: tiny, base, small, medium, large, turbo (default)")
        print("Example: python transcribe.py audio.mp3 turbo")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "turbo"
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    try:
        text = transcribe_audio(audio_file, model_size)
        
        # Print transcription
        print("\n" + "="*50)
        print("TRANSCRIPTION:")
        print("="*50)
        print(text)
        print("="*50)
        
        # Save to file
        output_file = Path(audio_file).stem + "_transcription.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"\nTranscription saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()