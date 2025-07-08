import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import threading
import queue
import vosk
import sounddevice as sd
import cv2
import numpy as np
import tensorflow as tf
import cairo
import pango
import pangocairo
from py_svg2gcode import generate_gcode, config
import serial
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel, ValidationError
import ast

# Global variables
text_queue = queue.Queue()  # Queue for text from voice and gesture inputs
stop_event = threading.Event()  # For graceful shutdown
svg_lock = threading.Lock()  # Lock for SVG generation
text_lock = threading.Lock()  # Lock for accumulated text
last_gesture = ""  # Track last recognized gesture
current_Y = 0  # Starting Y position in mm
accumulated_text = []  # Multi-line text for SVG
serial_conn = None  # Serial connection

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("writing_machine.log", maxBytes=1024*1024, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WritingMachine")

# Configuration schema using Pydantic
class Config(BaseModel):
    vosk_model_path: str
    isl_model_path: str
    serial_port: str
    baud_rate: int
    writing_area: list[int]
    line_spacing: float
    grammar_words: list[str]

def load_config():
    """Load and validate configuration from config.json."""
    try:
        with open("config.json", "r") as f:
            data = json.load(f)
        config = Config(**data)
        return config
    except FileNotFoundError:
        logger.error("config.json not found")
        raise
    except ValidationError as e:
        logger.error(f"Invalid config.json: {e}")
        raise
    except Exception as e:
        logger.error(f"Config loading error: {e}")
        raise

# Load configuration and models
try:
    config_data = load_config()
    VOSK_MODEL_PATH = config_data.vosk_model_path
    ISL_MODEL_PATH = config_data.isl_model_path
    SERIAL_PORT = config_data.serial_port
    BAUD_RATE = config_data.baud_rate
    WRITING_AREA = tuple(config_data.writing_area)
    LINE_SPACING = config_data.line_spacing
    GRAMMAR_WORDS = config_data.grammar_words
except Exception:
    logger.critical("Failed to load configuration. Exiting.")
    exit(1)

try:
    vosk_model = vosk.Model(VOSK_MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load Vosk model: {e}")
    exit(1)

try:
    interpreter = tf.lite.Interpreter(model_path=ISL_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    logger.error(f"Failed to load ISL model: {e}")
    exit(1)

# Gesture class names for ISL
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
               "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space"]

# GUI setup
root = ThemedTk(theme="arc")
root.title("Writing Machine Control")
root.geometry("500x600")

# GUI Elements
ttk.Label(root, text="Font:").grid(row=0, column=0, padx=5, pady=5)
font_var = tk.StringVar(value="Arial")
font_dropdown = ttk.Combobox(root, textvariable=font_var)
font_dropdown.grid(row=0, column=1, padx=5, pady=5)
ttk.Label(root, text="Size:").grid(row=1, column=0, padx=5, pady=5)
size_entry = ttk.Entry(root); size_entry.insert(0, "12"); size_entry.grid(row=1, column=1)
ttk.Label(root, text="Speed:").grid(row=2, column=0, padx=5, pady=5)
speed_entry = ttk.Entry(root); speed_entry.insert(0, "1000"); speed_entry.grid(row=2, column=1)
ttk.Label(root, text="Pen Up:").grid(row=3, column=0, padx=5, pady=5)
up_entry = ttk.Entry(root); up_entry.insert(0, "0"); up_entry.grid(row=3, column=1)
ttk.Label(root, text="Pen Down:").grid(row=4, column=0, padx=5, pady=5)
down_entry = ttk.Entry(root); down_entry.insert(0, "255"); down_entry.grid(row=4, column=1)
status_label = ttk.Label(root, text="Status: Ready"); status_label.grid(row=5, column=0, columnspan=2)
mode_label = ttk.Label(root, text="Mode: Idle"); mode_label.grid(row=6, column=0, columnspan=2)
preview_text = tk.Text(root, height=5, width=50); preview_text.grid(row=8, column=0, columnspan=2)
last_written_text = ttk.Label(root, text=""); last_written_text.grid(row=10, column=0, columnspan=2)

# Force Write function and button
def force_write():
    """Manually trigger writing of accumulated text."""
    global accumulated_text, current_Y
    with text_lock:
        if accumulated_text:
            height_mm = generate_svg(accumulated_text, font_var.get(), int(size_entry.get()), WRITING_AREA[0], current_Y)
            if height_mm > 0:
                generate_gcode_func("output.svg", "output.nc", int(up_entry.get()), int(down_entry.get()), int(speed_entry.get()))
                send_gcode("output.nc")
                current_Y += height_mm
                accumulated_text.clear()
                if current_Y > WRITING_AREA[1]:
                    status_label.config(text="Status: Page full")
                    messagebox.showinfo("Page Full", "Please insert a new page.")
                    current_Y = 0
                else:
                    status_label.config(text="Status: Writing complete")

force_write_button = ttk.Button(root, text="Force Write", command=force_write)
force_write_button.grid(row=12, column=0, columnspan=2, pady=10)

# Start and stop functions
def start_writing():
    """Start voice and gesture recognition and writing process."""
    try:
        stop_event.clear()
        toggle_inputs(False)
        status_label.config(text="Status: Writing...")
        threading.Thread(target=voice_recognition, daemon=True).start()
        threading.Thread(target=gesture_recognition, daemon=True).start()
        root.after(100, process_text)
    except ValueError as e:
        status_label.config(text=f"Error: Invalid input ({e})")
        messagebox.showerror("Input Error", str(e))

def stop_writing():
    """Stop all operations gracefully."""
    stop_event.set()
    toggle_inputs(True)
    status_label.config(text="Status: Stopped")
    mode_label.config(text="Mode: Idle")
    preview_text.delete(1.0, tk.END)

start_button = ttk.Button(root, text="Start Writing", command=start_writing)
start_button.grid(row=11, column=0, padx=5, pady=10)
stop_button = ttk.Button(root, text="Stop", command=stop_writing)
stop_button.grid(row=11, column=1, padx=5, pady=10)

def toggle_inputs(enabled):
    """Enable or disable input fields."""
    state = 'normal' if enabled else 'disabled'
    font_dropdown.config(state=state)
    size_entry.config(state=state)
    speed_entry.config(state=state)
    up_entry.config(state=state)
    down_entry.config(state=state)
    force_write_button.config(state=state)

# Voice recognition function
def voice_recognition():
    """Handle voice input with confirmation."""
    current_state = "listening"
    transcribed_text = ""
    grammar = json.dumps({"words": GRAMMAR_WORDS})
    try:
        rec = vosk.KaldiRecognizer(vosk_model, 16000, grammar)
        q = queue.Queue()

        def callback(indata, frames, time_info, status):
            q.put(bytes(indata))

        with sd.RawInputStream(samplerate=16000, channels=1, dtype='int16', callback=callback):
            while not stop_event.is_set():
                try:
                    data = q.get(timeout=0.1)
                    if current_state == "listening":
                        mode_label.config(text="Mode: Listening")
                        if rec.AcceptWaveform(data):
                            result = ast.literal_eval(rec.Result())['text']
                            if result:
                                transcribed_text = result
                                preview_text.delete(1.0, tk.END)
                                preview_text.insert(tk.END, f"{time.asctime()}: {result}")
                                current_state = "confirming"
                    elif current_state == "confirming":
                        mode_label.config(text="Mode: Confirming")
                        if rec.AcceptWaveform(data):
                            response = ast.literal_eval(rec.Result())['text'].lower()
                            if response == "yes":
                                with text_lock:
                                    text_queue.put(transcribed_text)
                                current_state = "listening"
                                preview_text.delete(1.0, tk.END)
                            elif response == "no" or response == "repeat":
                                current_state = "listening"
                                preview_text.delete(1.0, tk.END)
                except queue.Empty:
                    continue
    except Exception as e:
        logger.error(f"Voice recognition error: {e}")
        root.after(0, lambda: status_label.config(text=f"Voice Error: {e}"))
        root.after(0, lambda: messagebox.showerror("Voice Error", str(e)))

# Gesture recognition function
def gesture_recognition():
    """Handle gesture input from webcam using ISL model."""
    global last_gesture
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            # Preprocess frame for model (assumes 224x224 input)
            frame = cv2.resize(frame, (224, 224)) / 255.0
            interpreter.set_tensor(input_details[0]['index'], frame.reshape(1, 224, 224, 3).astype(np.float32))
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]
            confidence = np.max(prediction)
            if confidence > 0.8:  # Confidence threshold
                gesture = class_names[np.argmax(prediction)]
                if gesture != last_gesture:
                    with text_lock:
                        text_queue.put(" " if gesture == "space" else gesture)
                    last_gesture = gesture
            time.sleep(0.1)  # Control frame processing rate
    except Exception as e:
        logger.error(f"Gesture recognition error: {e}")
        root.after(0, lambda: status_label.config(text=f"Gesture Error: {e}"))
        root.after(0, lambda: messagebox.showerror("Gesture Error", str(e)))
    finally:
        if cap:
            cap.release()

# SVG generation function
def generate_svg(text_lines, font_name, font_size, width_mm, start_y):
    """Generate SVG from text with proper scaling."""
    with svg_lock:
        try:
            surface = cairo.SVGSurface("output.svg", WRITING_AREA[0], WRITING_AREA[1])
            surface.set_document_unit(cairo.SVGUnit.MM)
            cr = cairo.Context(surface)
            cr.scale(72 / 25.4, 72 / 25.4)  # Convert mm to points
            y_pos = start_y
            total_height = 0
            for text in text_lines:
                layout = pangocairo.create_layout(cr)
                layout.set_text(text, -1)
                layout.set_font_description(pango.FontDescription(f"{font_name} {font_size}"))
                layout.set_width(int(width_mm * (72 / 25.4) * 1024))  # Pango units
                layout.set_wrap(pango.WrapMode.WORD)
                cr.move_to(0, y_pos)
                pangocairo.show_layout(cr, layout)
                _, height = layout.get_size()
                height_mm = height / 1024.0 / (72 / 25.4)  # Convert back to mm
                y_pos += height_mm + LINE_SPACING
                total_height += height_mm + LINE_SPACING
            surface.finish()
            return total_height
        except Exception as e:
            logger.error(f"SVG generation error: {e}")
            root.after(0, lambda: status_label.config(text=f"SVG Error: {e}"))
            root.after(0, lambda: messagebox.showerror("SVG Error", str(e)))
            return 0

# G-code generation function
def generate_gcode_func(svg_file, gcode_file, pen_up, pen_down, speed):
    """Convert SVG to G-code with specified parameters."""
    try:
        config.pen_up_command = f"M3 S{pen_up}"
        config.pen_down_command = f"M3 S{pen_down}"
        config.feedrate = speed
        config.bed_size = WRITING_AREA
        generate_gcode(svg_file, gcode_file)
    except Exception as e:
        logger.error(f"G-code generation error: {e}")
        root.after(0, lambda: status_label.config(text=f"G-code Error: {e}"))
        root.after(0, lambda: messagebox.showerror("G-code Error", str(e)))

# Serial communication function
def send_gcode(gcode_file):
    """Send G-code to the writing machine with retries."""
    global serial_conn
    try:
        if not serial_conn or not serial_conn.is_open:
            serial_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
        time.sleep(2)  # Wait for connection
        with open(gcode_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if stop_event.is_set():
                    break
                retries = 0
                max_retries = 3
                while retries < max_retries:
                    try:
                        serial_conn.write((line.strip() + "\n").encode())
                        response = serial_conn.read_until(b'\n').decode().strip()
                        if response == "ok":
                            root.after(0, lambda i=i: status_label.config(text=f"Status: Writing line {i+1}/{len(lines)}"))
                            break
                        if "error" in response:
                            raise serial.SerialException(response)
                        time.sleep(2 ** retries)  # Exponential backoff
                        retries += 1
                    except serial.SerialException as e:
                        logger.warning(f"Serial retry {retries}: {e}")
                        if retries == max_retries - 1:
                            raise
    except Exception as e:
        logger.error(f"Serial communication error: {e}")
        root.after(0, lambda: status_label.config(text=f"Serial Error: {e}"))
        root.after(0, lambda: messagebox.showerror("Serial Error", str(e)))
    finally:
        if serial_conn and serial_conn.is_open:
            serial_conn.close()
            serial_conn = None

# Process text from queue
def process_text():
    """Process text from the queue and update GUI."""
    try:
        while not text_queue.empty():
            text = text_queue.get()
            with text_lock:
                accumulated_text.append(text)
            root.after(0, lambda t=text: last_written_text.config(text=f"Last: {t}"))
        if not stop_event.is_set():
            root.after(100, process_text)
    except Exception as e:
        logger.error(f"Text processing error: {e}")

# Cleanup on close
def on_close():
    """Handle application shutdown."""
    stop_event.set()
    if serial_conn and serial_conn.is_open:
        serial_conn.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# Populate font dropdown
def populate_fonts():
    """Populate font dropdown with system fonts."""
    pango_context = pango.Context()
    font_families = [family.get_name() for family in pango_context.list_families()]
    font_dropdown['values'] = font_families

populate_fonts()

# Run the GUI
if _name_ == "_main_":
    try:
        root.mainloop()
    except Exception as e:
        logger.error(f"GUI error: {e}")
        messagebox.showerror("GUI Error", str(e))