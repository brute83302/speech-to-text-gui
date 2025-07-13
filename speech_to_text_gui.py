#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from faster_whisper import WhisperModel, BatchedInferencePipeline
import threading
import os
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import tempfile
import time
import numpy as np
import gc
import logging

class SpeechToTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech to Text - Whisper")
        self.root.geometry("900x700")
        
        # Modern dark theme colors
        self.colors = {
            'bg': '#1a1a1a',
            'surface': '#2d2d2d', 
            'primary': '#6366f1',
            'primary_hover': '#4f46e5',
            'text': '#ffffff',
            'text_secondary': '#a1a1aa',
            'border': '#404040',
            'success': '#10b981',
            'error': '#ef4444',
            'warning': '#f59e0b'
        }
        
        self.setup_theme()
        
        self.model = None
        self.batched_model = None
        self.model_size = "turbo"
        self.model_loaded = False
        self.model_loading = False
        self.transcribing = False  # Flag to disable animations during transcription
        
        # Recording state
        self.is_recording = False
        self.recorded_file = None
        self.recording_data = []
        self.recording_chunks = []
        self.sample_rate = 16000
        self.sensitivity_threshold = 0.0001  # Much lower threshold
        self.current_level = 0.0
        self.level_update_counter = 0  # Throttle UI updates
        
        self.setup_ui()
        
        # Start model preloading in background
        self.start_model_preloading()
        
        # Setup cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_theme(self):
        """Setup modern dark theme"""
        self.root.configure(bg=self.colors['bg'])
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles with modern colors
        style.configure('Modern.TFrame', 
                       background=self.colors['bg'], 
                       borderwidth=0)
        
        style.configure('Surface.TFrame', 
                       background=self.colors['surface'], 
                       relief='flat',
                       borderwidth=1)
        
        style.configure('Modern.TLabel', 
                       background=self.colors['bg'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 10))
        
        style.configure('Surface.TLabel', 
                       background=self.colors['surface'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 10))
        
        style.configure('Title.TLabel', 
                       background=self.colors['bg'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 14, 'bold'))
        
        style.configure('Modern.TEntry', 
                       fieldbackground=self.colors['surface'],
                       foreground=self.colors['text'],
                       borderwidth=1,
                       relief='flat',
                       insertcolor=self.colors['text'])
        
        style.configure('Modern.TCombobox', 
                       fieldbackground=self.colors['surface'],
                       foreground=self.colors['text'],
                       background=self.colors['surface'],
                       borderwidth=1,
                       relief='flat')
        
        style.configure('Primary.TButton', 
                       background=self.colors['primary'],
                       foreground='white',
                       borderwidth=0,
                       relief='flat',
                       font=('Segoe UI', 10, 'bold'))
        
        style.map('Primary.TButton',
                 background=[('active', self.colors['primary_hover']),
                           ('pressed', self.colors['primary_hover'])])
        
        style.configure('Secondary.TButton', 
                       background=self.colors['surface'],
                       foreground=self.colors['text'],
                       borderwidth=1,
                       relief='flat',
                       font=('Segoe UI', 9))
        
        style.map('Secondary.TButton',
                 background=[('active', self.colors['border']),
                           ('pressed', self.colors['border'])])
        
        style.configure('Modern.Horizontal.TProgressbar',
                       background=self.colors['primary'],
                       troughcolor=self.colors['surface'],
                       borderwidth=0,
                       relief='flat')
        
        # Recording button styles
        style.configure('Recording.TButton', 
                       background=self.colors['error'],
                       foreground='white',
                       borderwidth=0,
                       relief='flat',
                       font=('Segoe UI', 10, 'bold'))
        
        style.map('Recording.TButton',
                 background=[('active', '#dc2626'),
                           ('pressed', '#dc2626')])
        
    def setup_ui(self):
        # Main container with modern styling
        main_frame = ttk.Frame(self.root, style='Modern.TFrame', padding="20")
        main_frame.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Speech to Text", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Model selection section
        model_frame = ttk.Frame(main_frame, style='Surface.TFrame', padding="15")
        model_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(0, 15))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model Size:", style='Surface.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="turbo")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                  values=["tiny", "base", "small", "medium", "large", "turbo"],
                                  style='Modern.TCombobox', state='readonly')
        model_combo.grid(row=0, column=1, sticky=tk.W+tk.E, pady=5, padx=(15, 0))
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # File selection section
        file_section = ttk.Frame(main_frame, style='Surface.TFrame', padding="15")
        file_section.grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(0, 15))
        file_section.columnconfigure(1, weight=1)
        
        ttk.Label(file_section, text="Audio File:", style='Surface.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        file_input_frame = ttk.Frame(file_section, style='Surface.TFrame')
        file_input_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E)
        file_input_frame.columnconfigure(0, weight=1)
        
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_input_frame, textvariable=self.file_var, 
                                   style='Modern.TEntry', font=('Segoe UI', 10))
        self.file_entry.grid(row=0, column=0, sticky=tk.W+tk.E, padx=(0, 10))
        
        browse_btn = ttk.Button(file_input_frame, text="Browse", command=self.browse_file,
                               style='Secondary.TButton')
        browse_btn.grid(row=0, column=1, padx=(0, 10))
        
        # Microphone recording button
        self.mic_btn = ttk.Button(file_input_frame, text="🎤 Record", command=self.toggle_recording,
                                 style='Secondary.TButton')
        self.mic_btn.grid(row=0, column=2)
        self.add_button_hover_effect(self.mic_btn)
        
        # Action buttons section
        action_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        action_frame.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        self.transcribe_btn = ttk.Button(action_frame, text="Start Transcription", 
                                        command=self.transcribe_file, style='Primary.TButton')
        self.transcribe_btn.pack(pady=10)
        
        # Add hover effect animation
        self.add_button_hover_effect(self.transcribe_btn)
        
        # Test microphone button
        test_mic_btn = ttk.Button(action_frame, text="Test Microphone", 
                                 command=self.test_microphone, style='Secondary.TButton')
        test_mic_btn.pack(pady=(5, 0))
        self.add_button_hover_effect(test_mic_btn)
        
        # Sensitivity controls
        sensitivity_frame = ttk.Frame(action_frame, style='Modern.TFrame')
        sensitivity_frame.pack(pady=(10, 0))
        
        ttk.Label(sensitivity_frame, text="Sensitivity:", style='Modern.TLabel', font=('Segoe UI', 9)).pack(side=tk.LEFT)
        
        sensitivity_buttons = ttk.Frame(sensitivity_frame, style='Modern.TFrame')
        sensitivity_buttons.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(sensitivity_buttons, text="Low", command=lambda: self.set_sensitivity(0.001), 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(sensitivity_buttons, text="Normal", command=lambda: self.set_sensitivity(0.0001), 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(sensitivity_buttons, text="High", command=lambda: self.set_sensitivity(0.00001), 
                  style='Secondary.TButton').pack(side=tk.LEFT)
        
        # Progress and status section
        progress_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(0, 15))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate', 
                                       style='Modern.Horizontal.TProgressbar')
        self.progress.grid(row=0, column=0, sticky=tk.W+tk.E, pady=(0, 5))
        
        # Audio level indicator
        level_frame = ttk.Frame(progress_frame, style='Modern.TFrame')
        level_frame.grid(row=1, column=0, sticky=tk.W+tk.E, pady=(5, 0))
        level_frame.columnconfigure(1, weight=1)
        
        ttk.Label(level_frame, text="Audio Level:", style='Modern.TLabel', font=('Segoe UI', 8)).grid(row=0, column=0, sticky=tk.W)
        
        self.level_progress = ttk.Progressbar(level_frame, mode='determinate', 
                                             style='Modern.Horizontal.TProgressbar', length=200)
        self.level_progress.grid(row=0, column=1, sticky=tk.W+tk.E, padx=(10, 10))
        
        self.level_label = ttk.Label(level_frame, text="0%", style='Modern.TLabel', font=('Segoe UI', 8))
        self.level_label.grid(row=0, column=2, sticky=tk.W)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var, 
                                     style='Modern.TLabel', font=('Segoe UI', 9))
        self.status_label.grid(row=2, column=0, pady=5)
        
        # Transcription output section
        output_frame = ttk.Frame(main_frame, style='Surface.TFrame', padding="15")
        output_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S, pady=(0, 0))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Transcription Result:", 
                 style='Surface.TLabel', font=('Segoe UI', 11, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Text output with modern styling
        self.text_output = scrolledtext.ScrolledText(
            output_frame, 
            wrap=tk.WORD, 
            height=12,
            bg=self.colors['bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            selectbackground=self.colors['primary'],
            selectforeground='white',
            font=('Segoe UI', 10),
            relief='flat',
            borderwidth=1
        )
        self.text_output.grid(row=1, column=0, sticky=tk.W+tk.E+tk.N+tk.S, pady=(0, 15))
        
        # Button frame for save and copy
        button_frame = ttk.Frame(output_frame, style='Surface.TFrame')
        button_frame.grid(row=2, column=0, pady=(0, 0))
        
        self.save_btn = ttk.Button(button_frame, text="Save Transcription", 
                                  command=self.save_transcription, state='disabled',
                                  style='Secondary.TButton')
        self.save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.copy_btn = ttk.Button(button_frame, text="Copy to Clipboard", 
                                  command=self.copy_to_clipboard, state='disabled',
                                  style='Secondary.TButton')
        self.copy_btn.pack(side=tk.LEFT)
        
        # Drag and drop setup
        self.setup_drag_drop()
        
        # Add fade-in animation
        self.fade_in_animation()
    
    def add_button_hover_effect(self, button):
        """Add subtle hover animation to buttons"""
        def on_enter(e):
            button.configure(cursor='hand2')
        
        def on_leave(e):
            button.configure(cursor='')
        
        button.bind('<Enter>', on_enter)
        button.bind('<Leave>', on_leave)
    
    def on_closing(self):
        """Handle application closing with proper cleanup"""
        try:
            # Stop any ongoing recording
            if self.is_recording:
                self.is_recording = False
                try:
                    sd.stop()
                except:
                    pass
            
            # Clean up temporary files
            if hasattr(self, 'recorded_file') and self.recorded_file and os.path.exists(self.recorded_file):
                try:
                    os.unlink(self.recorded_file)
                except:
                    pass
            
            # Clear recording data
            if hasattr(self, 'recording_data'):
                self.recording_data = None
            if hasattr(self, 'recording_chunks'):
                self.recording_chunks.clear()
            
            # Clear models to free GPU/CPU memory
            if hasattr(self, 'model'):
                self.model = None
            if hasattr(self, 'batched_model'):
                self.batched_model = None
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Cleanup error: {e}")
        finally:
            self.root.quit()
            self.root.destroy()
    
    def fade_in_animation(self):
        """Subtle fade-in animation for the main window"""
        if not self.transcribing:  # Skip animation if transcribing
            self.root.attributes('-alpha', 0.0)
            self.root.after(50, self._fade_in_step, 0.0)
        else:
            self.root.attributes('-alpha', 1.0)
    
    def _fade_in_step(self, alpha):
        """Step function for fade-in animation"""
        if not self.transcribing:  # Don't animate during transcription
            alpha += 0.1
            self.root.attributes('-alpha', alpha)
            if alpha < 1.0:
                self.root.after(50, self._fade_in_step, alpha)
        else:
            self.root.attributes('-alpha', 1.0)
    
    def animate_progress_start(self):
        """Smooth progress bar animation (optimized)"""
        if not self.transcribing:
            self.progress.start(10)  # Faster, smoother animation
        else:
            self.progress.start(30)  # Slower during transcription to save CPU
    
    def animate_progress_stop(self):
        """Stop progress bar animation smoothly"""
        self.progress.stop()
        # Force a UI update to clear progress
        self.root.update_idletasks()
        
    def setup_drag_drop(self):
        """Setup drag and drop functionality (placeholder for future implementation)"""
        # Note: Full drag-and-drop requires additional libraries like tkinterdnd2
        # For now, we'll skip this feature to avoid dependencies
        pass
    
    def browse_file(self):
        """Open file dialog to select audio file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.flac *.m4a *.ogg *.wma"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.file_var.set(file_path)
    
    def on_model_change(self, event=None):
        """Handle model size change"""
        self.model_size = self.model_var.get()
        self.model = None  # Reset model to reload with new size
        self.batched_model = None
        self.model_loaded = False
        self.status_var.set(f"Model changed to {self.model_size}")
        # Start preloading new model in background
        self.start_model_preloading()
    
    def start_model_preloading(self):
        """Start model preloading in background thread"""
        if not self.model_loading:
            self.model_loading = True
            thread = threading.Thread(target=self._preload_model_worker)
            thread.daemon = True
            thread.start()
    
    def _preload_model_worker(self):
        """Background worker to preload model"""
        try:
            self.root.after(0, lambda: self.status_var.set(f"Preloading {self.model_size} model..."))
            
            # Determine optimal device and compute type
            device = "auto"
            compute_type = "auto"
            
            # Optimize compute type based on available hardware
            device = "cpu"  # Default to CPU
            compute_type = "int8"  # Default to compatible CPU type
            
            try:
                import torch
                
                # Check for NVIDIA GPU (CUDA)
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                    print("Using NVIDIA GPU (CUDA)")
                
                # Check for AMD GPU (ROCm/HIP)
                elif hasattr(torch, 'hip') and torch.hip.is_available():
                    device = "hip"
                    compute_type = "float16"
                    print("Using AMD GPU (ROCm)")
                
                # Check for OpenVINO GPU support (good for Intel/AMD integrated GPUs)
                else:
                    try:
                        import openvino as ov
                        core = ov.Core()
                        available_devices = core.available_devices
                        
                        # Check for GPU device
                        if "GPU" in available_devices:
                            print(f"OpenVINO GPU available on devices: {available_devices}")
                            device = "auto"  # Let faster-whisper + OpenVINO optimize
                            compute_type = "int8"  # Good balance for integrated GPU
                        else:
                            print(f"OpenVINO devices available: {available_devices}")
                            device = "cpu"
                            compute_type = "int8"
                    except ImportError:
                        # Fallback: Try faster-whisper auto detection
                        try:
                            from faster_whisper import WhisperModel
                            test_model = WhisperModel('tiny', device='auto', compute_type='auto')
                            device = "auto"
                            compute_type = "auto"
                            print("Using faster-whisper auto GPU detection")
                            del test_model  # cleanup
                        except Exception:
                            print("No GPU acceleration available, using optimized CPU")
                            device = "cpu"
                            compute_type = "int8"
                        
            except ImportError:
                print("PyTorch not available, using CPU with auto detection")
                device = "auto"
                compute_type = "auto"
            
            # Load base model for BatchedInferencePipeline with fallback
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                    num_workers=2  # Optimize for batch processing
                )
            except Exception as model_error:
                # Fallback to default compute type if optimized type fails
                print(f"Optimized compute type failed, trying default: {model_error}")
                self.model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type="default",
                    num_workers=2
                )
            
            # Create BatchedInferencePipeline for faster processing
            self.batched_model = BatchedInferencePipeline(model=self.model)
            
            self.model_loaded = True
            self.model_loading = False
            
            # Determine device name for display
            if device == "cuda":
                device_name = "NVIDIA GPU"
            elif device == "hip":
                device_name = "AMD GPU"
            elif device == "auto":
                device_name = "Auto GPU"
            else:
                device_name = "CPU"
            
            print(f"Model loaded: {self.model_size} on {device_name} with {compute_type} precision")
            
            self.root.after(0, lambda: self.status_var.set(
                f"{self.model_size.title()} model ready ({device_name}, {compute_type})"
            ))
            
        except Exception as e:
            self.model_loading = False
            error_msg = str(e)[:50]
            self.root.after(0, lambda msg=error_msg: self.status_var.set(f"Model loading failed: {msg}..."))
            print(f"Model preloading error: {e}")
    
    def load_model(self):
        """Load Whisper model with GPU acceleration (legacy method)"""
        if not self.model_loaded:
            if not self.model_loading:
                self.start_model_preloading()
            # Wait for model to load
            while self.model_loading and not self.model_loaded:
                time.sleep(0.1)
                self.root.update()
    
    def transcribe_file(self):
        """Transcribe the selected audio file with optimized UI"""
        file_path = self.file_var.get().strip()
        
        if not file_path:
            messagebox.showerror("Error", "Please select an audio file first!")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return
        
        # Set transcribing flag to disable animations
        self.transcribing = True
        
        # Disable button and start progress (no animation during transcription)
        self.transcribe_btn.config(state='disabled')
        self.progress.start(20)  # Faster, less CPU-intensive progress
        self.text_output.delete(1.0, tk.END)
        self.save_btn.config(state='disabled')
        
        # Direct status update (no animation)
        self.status_var.set("Preparing transcription...")
        self.root.update_idletasks()  # Update UI once
        
        # Run transcription in separate thread
        thread = threading.Thread(target=self._transcribe_worker, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def _transcribe_worker(self, file_path):
        """Worker function for transcription (runs in separate thread)"""
        try:
            # Ensure model is loaded
            self.load_model()
            
            # Check if model is loaded
            if not self.model_loaded or (self.model is None and self.batched_model is None):
                self.root.after(0, self._transcription_error, "Failed to load model")
                return
            
            # Update status
            self.root.after(0, lambda: self.status_var.set("Transcribing audio..."))
            
            # Use BatchedInferencePipeline if available for 3x speedup
            if self.batched_model is not None:
                try:
                    # BatchedInferencePipeline approach (faster)
                    segments, info = self.batched_model.transcribe(file_path)
                    transcription = " ".join([segment.text for segment in segments])
                except Exception as batch_error:
                    print(f"Batch processing failed, falling back to regular: {batch_error}")
                    # Fallback to regular model
                    segments, info = self.model.transcribe(file_path, beam_size=1)  # Faster beam size
                    transcription = " ".join([segment.text for segment in segments])
            else:
                # Regular model approach with optimized settings
                segments, info = self.model.transcribe(
                    file_path,
                    beam_size=1,  # Faster than default beam_size=5
                    best_of=1,    # Faster than default best_of=5
                    temperature=0.0  # Deterministic for speed
                )
                transcription = " ".join([segment.text for segment in segments])
            
            # Clean up memory
            if segments:
                del segments
            gc.collect()
            
            # Update UI in main thread
            self.root.after(0, self._transcription_complete, transcription)
            
        except Exception as e:
            self.root.after(0, self._transcription_error, str(e))
    
    def _transcription_complete(self, transcription):
        """Handle successful transcription completion with optimized UI"""
        self.transcribing = False  # Re-enable animations
        self.progress.stop()
        self.transcribe_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.copy_btn.config(state='normal')
        
        # Direct text insertion (no animation for speed)
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, transcription)
        
        # Direct status update
        self.status_var.set("Transcription complete!")
        
        # Clean up temporary recording file
        if self.recorded_file and os.path.exists(self.recorded_file):
            try:
                os.unlink(self.recorded_file)
                self.recorded_file = None
            except:
                pass
                
        # Force garbage collection after transcription
        gc.collect()
    
    def animate_text_insertion(self, text):
        """Animate text insertion with typing effect (disabled during transcription)"""
        if self.transcribing:
            # During transcription, insert directly for performance
            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(tk.END, text)
        else:
            # Normal animation when not transcribing
            self.text_output.delete(1.0, tk.END)
            self._typing_animation(text, 0)
    
    def _typing_animation(self, text, index):
        """Step function for typing animation (optimized)"""
        if not self.transcribing and index < len(text):
            self.text_output.insert(tk.END, text[index])
            self.text_output.see(tk.END)
            # Faster typing for longer texts, skip animation if transcribing
            delay = 1 if len(text) > 500 else 2  # Slightly faster
            self.root.after(delay, self._typing_animation, text, index + 1)
        elif self.transcribing:
            # If transcribing started during animation, just insert remaining text
            self.text_output.insert(tk.END, text[index:])
    
    def animate_status_change(self, new_status):
        """Animate status text changes (disabled during transcription)"""
        if self.transcribing:
            # Direct update during transcription for performance
            self.status_var.set(new_status)
        else:
            # Normal animation when not transcribing
            current_color = self.status_label.cget('foreground')
            self._fade_status_out(new_status, 1.0)
    
    def _fade_status_out(self, new_status, alpha):
        """Fade out current status text (skip during transcription)"""
        if not self.transcribing:
            alpha -= 0.2
            if alpha <= 0:
                self.status_var.set(new_status)
                self._fade_status_in(1.0)
            else:
                self.root.after(30, self._fade_status_out, new_status, alpha)
        else:
            self.status_var.set(new_status)
    
    def _fade_status_in(self, alpha):
        """Fade in new status text (skip during transcription)"""
        if not self.transcribing:
            alpha -= 0.2
            if alpha > 0:
                self.root.after(30, self._fade_status_in, alpha)
    
    def _transcription_error(self, error_msg):
        """Handle transcription error with optimized cleanup"""
        self.transcribing = False  # Re-enable animations
        self.progress.stop()
        self.transcribe_btn.config(state='normal')
        
        # Direct status update
        self.status_var.set("Error during transcription")
        messagebox.showerror("Transcription Error", f"Error: {error_msg}")
        
        # Cleanup memory on error
        gc.collect()
    
    def toggle_recording(self):
        """Toggle microphone recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording from microphone"""
        try:
            # Check for available input devices
            devices = sd.query_devices()
            input_devices = []
            for d in devices:
                try:
                    if d['max_input_channels'] > 0:
                        input_devices.append(d)
                except (KeyError, TypeError):
                    continue
            
            if not input_devices:
                messagebox.showerror("No Microphone", "No input devices found. Please check your microphone connection.")
                return
            
            # Use default input device
            default_device = sd.default.device[0]  # Input device
            device_info = sd.query_devices(default_device)
            
            self.is_recording = True
            self.recording_data = []
            
            # Update UI
            self.mic_btn.configure(text="⏹️ Stop", style='Recording.TButton')
            try:
                device_name = device_info['name']
            except (KeyError, TypeError):
                device_name = 'Unknown Device'
            self.animate_status_change(f"Recording from {device_name}... Click stop when finished")
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
        except Exception as e:
            messagebox.showerror("Recording Error", f"Failed to start recording: {e}")
            self.is_recording = False
    
    def _record_audio(self):
        """Record audio in separate thread with optimized performance"""
        try:
            # Optimized recording settings
            self.sample_rate = 16000
            chunk_size = int(0.2 * self.sample_rate)  # 200ms chunks (reduced overhead)
            self.recording_chunks = []
            self.level_update_counter = 0
            
            print(f"Starting optimized recording with {chunk_size} sample chunks...")
            
            # Optimized audio callback with reduced overhead
            def audio_callback(indata, frames, time, status):
                if self.is_recording:
                    # Direct append without copy for performance
                    self.recording_chunks.append(indata.copy())
                    
                    # Heavily throttled UI updates to reduce overhead
                    self.level_update_counter += 1
                    if self.level_update_counter % 25 == 0:  # Update every 5 seconds (25 * 200ms)
                        current_level = np.max(np.abs(indata))
                        level_percent = min(100, current_level * 1000)
                        # Use after_idle to prevent UI blocking
                        self.root.after_idle(self.update_level_indicator, level_percent, current_level)
            
            # Optimized stream settings
            with sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=chunk_size,
                dtype=np.float32,
                latency='low'  # Optimize for low latency
            ):
                # Keep recording while flag is true
                while self.is_recording:
                    time.sleep(0.1)  # Larger sleep to reduce CPU usage
            
            # Efficient chunk combination
            if self.recording_chunks:
                self.recording_data = np.concatenate(self.recording_chunks, axis=0).flatten()
                duration = len(self.recording_data) / self.sample_rate
                print(f"Recorded {len(self.recording_data)} samples ({duration:.1f} seconds)")
                
                # Clear chunks to free memory immediately
                self.recording_chunks.clear()
            else:
                self.recording_data = np.array([])
                print("No data recorded")
                
        except Exception as e:
            print(f"Recording error: {e}")
            self.recording_data = np.array([])
            if hasattr(self, 'recording_chunks'):
                self.recording_chunks.clear()
            self.root.after(0, lambda: messagebox.showerror("Recording Error", f"Recording failed: {e}"))
    
    def stop_recording(self):
        """Stop recording and process audio with optimized cleanup"""
        self.is_recording = False
        
        # Wait for recording thread to finish cleanly
        time.sleep(0.2)
        
        # Stop sounddevice recording
        try:
            sd.stop()
        except:
            pass
        
        # Update UI
        self.mic_btn.configure(text="🎤 Record", style='Secondary.TButton')
        
        # Process recorded audio with better error handling
        try:
            if hasattr(self, 'recording_data') and len(self.recording_data) > 0:
                max_amplitude = np.max(np.abs(self.recording_data))
                duration = len(self.recording_data) / self.sample_rate
                print(f"Recording complete: {duration:.1f}s, max amplitude: {max_amplitude}")
                
                # Optimize audio for transcription (normalize if too quiet)
                if max_amplitude > 0:
                    # Normalize audio to improve transcription quality
                    if max_amplitude < 0.1:
                        self.recording_data = self.recording_data * (0.1 / max_amplitude)
                        print("Audio normalized for better transcription")
                
                # Save to optimized temporary file
                self.recorded_file = tempfile.mktemp(suffix=".wav")
                sf.write(self.recorded_file, self.recording_data, self.sample_rate, subtype='PCM_16')
                
                # Verify and process
                if os.path.exists(self.recorded_file):
                    file_size = os.path.getsize(self.recorded_file)
                    print(f"Audio saved: {file_size} bytes")
                    
                    self.file_var.set(self.recorded_file)
                    self.status_var.set(f"Recording ready ({duration:.1f}s), starting transcription...")
                    
                    # Clear recording data to free memory
                    self.recording_data = None
                    gc.collect()
                    
                    # Auto-start transcription with shorter delay
                    self.root.after(1000, self.transcribe_file)
                else:
                    raise Exception("Failed to create audio file")
            else:
                # Handle empty recording
                print("No audio data recorded")
                self.recording_data = np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)
                self.recorded_file = tempfile.mktemp(suffix=".wav")
                sf.write(self.recorded_file, self.recording_data, self.sample_rate)
                self.file_var.set(self.recorded_file)
                self.status_var.set("No audio detected, attempting transcription...")
                self.root.after(1000, self.transcribe_file)
                
        except Exception as e:
            print(f"Error processing recording: {e}")
            self.status_var.set(f"Recording error: {str(e)[:50]}...")
            messagebox.showerror("Recording Error", f"Failed to process recording: {e}")
            
            # Cleanup on error
            if hasattr(self, 'recording_data'):
                self.recording_data = None
            if hasattr(self, 'recording_chunks'):
                self.recording_chunks.clear()
            gc.collect()
    
    def update_level_indicator(self, level_percent, raw_level):
        """Update the audio level indicator (optimized)"""
        try:
            # Only update if widget still exists and we're recording
            if self.is_recording and hasattr(self, 'level_progress'):
                self.level_progress['value'] = level_percent
                self.level_label.config(text=f"{level_percent:.0f}%")
        except Exception:
            # Ignore errors if widgets are being destroyed
            pass
    
    def set_sensitivity(self, threshold):
        """Set microphone sensitivity threshold"""
        self.sensitivity_threshold = threshold
        self.animate_status_change(f"Sensitivity set to {threshold} (Lower = more sensitive)")
    
    def test_microphone(self):
        """Test if microphone is working"""
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
            
            if not input_devices:
                messagebox.showinfo("Microphone Test", "❌ No input devices found.\nPlease check your microphone connection.")
                return
            
            default_device = sd.default.device[0]
            device_info = sd.query_devices(default_device)
            
            # Test recording for 1 second
            test_duration = 1.0
            test_data = sd.rec(
                int(test_duration * 16000), 
                samplerate=16000, 
                channels=1, 
                dtype=np.float32,
                blocking=True
            )
            
            max_amplitude = np.max(np.abs(test_data))
            
            device_name = device_info.get('name', 'Unknown Device')
            device_samplerate = device_info.get('default_samplerate', 'Unknown')
            
            if max_amplitude > self.sensitivity_threshold:
                messagebox.showinfo("Microphone Test", 
                    f"✅ Microphone working!\n\n"
                    f"Device: {device_name}\n"
                    f"Sample Rate: {device_samplerate} Hz\n"
                    f"Signal Level: {max_amplitude:.6f}\n"
                    f"Threshold: {self.sensitivity_threshold:.6f}\n"
                    f"Status: ABOVE threshold ✓")
            else:
                messagebox.showwarning("Microphone Test", 
                    f"⚠️ Microphone detected but signal too quiet.\n\n"
                    f"Device: {device_name}\n"
                    f"Signal Level: {max_amplitude:.6f}\n"
                    f"Threshold: {self.sensitivity_threshold:.6f}\n"
                    f"Status: BELOW threshold ✗\n\n"
                    f"Try: Higher sensitivity or speak louder.")
                
        except Exception as e:
            messagebox.showerror("Microphone Test", f"❌ Microphone test failed:\n{e}")
    
    def copy_to_clipboard(self):
        """Copy transcription to clipboard"""
        transcription = self.text_output.get(1.0, tk.END).strip()
        if not transcription:
            messagebox.showwarning("Warning", "No transcription to copy!")
            return
        
        self.root.clipboard_clear()
        self.root.clipboard_append(transcription)
        self.animate_status_change("Transcription copied to clipboard!")
        
        # Show tooltip-like message
        self.show_copy_feedback()
    
    def show_copy_feedback(self):
        """Show brief feedback when text is copied"""
        # Create temporary label for feedback
        feedback = ttk.Label(self.root, text="✓ Copied to clipboard!", 
                            background=self.colors['success'], 
                            foreground='white',
                            font=('Segoe UI', 9))
        
        # Position near copy button
        x = self.root.winfo_x() + 50
        y = self.root.winfo_y() + self.root.winfo_height() - 100
        
        feedback.place(x=x, y=y)
        
        # Remove after 2 seconds
        self.root.after(2000, feedback.destroy)
    
    def save_transcription(self):
        """Save transcription to file"""
        transcription = self.text_output.get(1.0, tk.END).strip()
        if not transcription:
            messagebox.showwarning("Warning", "No transcription to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Transcription",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                messagebox.showinfo("Success", f"Transcription saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")

def main():
    root = tk.Tk()
    app = SpeechToTextApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()