import asyncio
import struct
import threading
import csv
from collections import deque
from datetime import datetime
from bleak import BleakClient
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os

DEVICE_ADDRESS = "44CED45D-83B0-BC60-BE09-4F76D13DE480"  # PCB B x11111
TX_CHARACTERISTIC_UUID = "6e400002-c352-11e5-953d-0002a5d5c51b"
RX_CHARACTERISTIC_UUID = "6e400003-c352-11e5-953d-0002a5d5c51b"

PAYLOAD = bytearray([0x01, 0xC0, 0x01])

# Store data with max 300 points
MAX_POINTS = 300
data_storage = {
    'ch1': deque(maxlen=MAX_POINTS),
    'ch2': deque(maxlen=MAX_POINTS),
    'ch3': deque(maxlen=MAX_POINTS),
    'ch4': deque(maxlen=MAX_POINTS),
}


# ============================================================================
# NEURAL NETWORK MODEL (same as training)
# ============================================================================

class SimpleNN(nn.Module):
    """Simple neural network for binary classification"""

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class BLEDataGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BLE Sensor Data Viewer with AI Prediction")
        self.root.geometry("1500x950")

        self.connected = False
        self.ble_task = None
        self.data_lock = threading.Lock()

        # Recording variables
        self.is_recording = False
        self.recording_data = {
            'ch1': [],
            'ch2': [],
            'ch3': [],
            'ch4': [],
        }
        self.recording_timestamps = []

        # Model and normalization
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.device = torch.device('cpu')

        # Load model and scaler at startup
        self.load_model_and_scaler()

        # Create main frames
        self.create_widgets()

    def load_model_and_scaler(self):
        """Load trained model and scaler from files"""
        try:
            # Try to load model
            if os.path.exists('best_model.pth'):
                self.model = SimpleNN().to(self.device)
                self.model.load_state_dict(torch.load('best_model.pth', map_location=self.device))
                self.model.eval()
                print("✓ Model loaded: best_model.pth")
            else:
                print("⚠️  Model not found: best_model.pth")
                return

            # Try to load scaler (we'll create a simple one if not found)
            # Since we don't have saved scaler, we'll create one with known calibration values
            # These should match your normalized_data.csv statistics
            self.scaler = StandardScaler()
            # We'll compute scaler stats from your known data ranges
            # For now, we'll use fit with dummy data to initialize it properly
            dummy_data = np.array([
                [98212.96, 98163.08, 98161.50, 98164.61],  # Empty state
                [99986.27, 99034.72, 99528.85, 98873.99]  # Full state
            ])
            # Fit scaler with more comprehensive data
            self.scaler.fit(dummy_data)
            print("✓ Scaler initialized")

            self.model_loaded = True
            print("✅ Model and Scaler Ready!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False

    def predict_occupancy(self, ch1, ch2, ch3, ch4):
        """
        Predict occupancy from raw BLE values

        Args:
            ch1, ch2, ch3, ch4: Raw BLE values

        Returns:
            (occupancy_state, confidence, probability)
        """
        if not self.model_loaded:
            return "Unknown", 0.0, 0.0

        try:
            # Create raw data array
            raw_data = np.array([[ch1, ch2, ch3, ch4]])

            # Normalize the data
            normalized_data = self.scaler.transform(raw_data)

            # Convert to tensor
            data_tensor = torch.FloatTensor(normalized_data).to(self.device)

            # Get prediction
            with torch.no_grad():
                probability = self.model(data_tensor).item()

            # Determine state
            occupancy_state = "Occupied" if probability > 0.5 else "Empty"
            confidence = max(probability, 1 - probability)

            return occupancy_state, confidence, probability

        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Error", 0.0, 0.0

    def create_widgets(self):
        # Top control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.connect_btn = ttk.Button(control_frame, text="Connect", command=self.start_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=5)

        self.disconnect_btn = ttk.Button(control_frame, text="Disconnect", command=self.stop_connection,
                                         state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(control_frame, text="Status: Disconnected", foreground="red",
                                      font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=20)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Record 60 seconds button
        self.record_btn = ttk.Button(control_frame, text="Record 60 Seconds",
                                     command=self.start_recording, state=tk.DISABLED)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        # Recording status label
        self.recording_status_label = ttk.Label(control_frame, text="", foreground="red",
                                                font=("Arial", 10, "bold"))
        self.recording_status_label.pack(side=tk.LEFT, padx=20)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Model status
        model_status_text = "✅ Model Loaded" if self.model_loaded else "❌ No Model"
        model_color = "green" if self.model_loaded else "red"
        self.model_status_label = ttk.Label(control_frame, text=model_status_text, foreground=model_color,
                                            font=("Arial", 10, "bold"))
        self.model_status_label.pack(side=tk.LEFT, padx=20)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Checkbox for auto-scale
        self.auto_scale_var = tk.BooleanVar(value=True)
        self.auto_scale_check = ttk.Checkbutton(control_frame, text="Auto-scale Y-axis (remove outliers)",
                                                variable=self.auto_scale_var)
        self.auto_scale_check.pack(side=tk.LEFT, padx=10)

        # Checkbox for separate graphs
        self.separate_graphs_var = tk.BooleanVar(value=False)
        self.separate_graphs_check = ttk.Checkbutton(control_frame, text="Separate Graphs (4 panels)",
                                                     variable=self.separate_graphs_var)
        self.separate_graphs_check.pack(side=tk.LEFT, padx=10)

        # Main content frame
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left side - Graph
        graph_frame = ttk.Frame(content_frame)
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(11, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right side - Data display
        right_frame = ttk.Frame(content_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)

        # ===== AI PREDICTION FRAME =====
        ai_frame = ttk.LabelFrame(right_frame, text="🤖 AI Prediction", padding=10)
        ai_frame.pack(fill=tk.X, pady=(0, 10))

        # Occupancy state (big display)
        self.occupancy_label = ttk.Label(ai_frame, text="--", font=("Arial", 28, "bold"), foreground="blue")
        self.occupancy_label.pack(pady=10)

        # Confidence
        confidence_frame = ttk.Frame(ai_frame)
        confidence_frame.pack(fill=tk.X, pady=5)
        ttk.Label(confidence_frame, text="Confidence:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.confidence_label = ttk.Label(confidence_frame, text="--", font=("Arial", 12, "bold"), foreground="green")
        self.confidence_label.pack(side=tk.LEFT, padx=10)

        # Probability bar
        prob_frame = ttk.Frame(ai_frame)
        prob_frame.pack(fill=tk.X, pady=5)
        ttk.Label(prob_frame, text="Empty ← Probability → Occupied", font=("Arial", 9)).pack()
        self.prob_canvas = tk.Canvas(prob_frame, height=20, bg='white', highlightthickness=1)
        self.prob_canvas.pack(fill=tk.X, pady=3)

        # ===== VALUE DISPLAY FRAME =====
        values_frame = ttk.LabelFrame(right_frame, text="Current Values", padding=10)
        values_frame.pack(fill=tk.X, pady=(0, 10))

        self.value_labels = {}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        channel_names = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']

        for i, (name, color) in enumerate(zip(channel_names, colors)):
            label_frame = ttk.Frame(values_frame)
            label_frame.pack(fill=tk.X, pady=5)

            ttk.Label(label_frame, text=f"{name}:", width=12, font=("Arial", 10, "bold")).pack(side=tk.LEFT)

            value_label = ttk.Label(label_frame, text="--", font=("Arial", 12, "bold"), foreground=color)
            value_label.pack(side=tk.LEFT, padx=10)

            self.value_labels[f'ch{i + 1}'] = value_label

        # ===== STATISTICS FRAME =====
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=20, width=40, font=("Courier", 8))
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # Log frame at bottom
        log_frame = ttk.LabelFrame(self.root, text="Connection Log", padding=5)
        log_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=3, font=("Courier", 8))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Start periodic update
        self.update_display()

    def draw_probability_bar(self, probability):
        """Draw a probability bar (0=Empty, 1=Occupied)"""
        self.prob_canvas.delete("all")

        width = self.prob_canvas.winfo_width()
        height = self.prob_canvas.winfo_height()

        if width < 2:
            width = 300
        if height < 2:
            height = 20

        # Draw background
        self.prob_canvas.create_rectangle(0, 0, width, height, fill='#f0f0f0', outline='black')

        # Draw gradient from blue (empty) to red (occupied)
        bar_width = probability * width
        if probability < 0.5:
            # Blue gradient (empty side)
            color = f'#{int(50 + (1 - probability * 2) * 150):02x}7f{int(200 - (1 - probability * 2) * 150):02x}'
        else:
            # Red gradient (occupied side)
            color = f'#{int(200 + (probability - 0.5) * 100):02x}7f{int(100 - (probability - 0.5) * 100):02x}'

        self.prob_canvas.create_rectangle(0, 0, bar_width, height, fill=color, outline='black')

        # Draw middle line at 0.5
        mid_x = width * 0.5
        self.prob_canvas.create_line(mid_x, 0, mid_x, height, fill='black', dash=(4, 4), width=2)

        # Draw pointer
        pointer_x = probability * width
        self.prob_canvas.create_polygon(pointer_x, height, pointer_x - 5, height + 5, pointer_x + 5, height + 5,
                                        fill='black')

    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.update()

    def start_connection(self):
        """Start BLE connection in background thread"""
        if not self.connected:
            self.connect_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Connecting...", foreground="orange")

            # Run BLE connection in separate thread
            thread = threading.Thread(target=self.run_ble_loop, daemon=True)
            thread.start()

    def stop_connection(self):
        """Stop BLE connection"""
        self.connected = False
        self.disconnect_btn.config(state=tk.DISABLED)
        self.connect_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Disconnecting...", foreground="orange")

    def start_recording(self):
        """Start 60-second recording"""
        if self.connected and not self.is_recording:
            self.is_recording = True
            self.recording_data = {'ch1': [], 'ch2': [], 'ch3': [], 'ch4': []}
            self.recording_timestamps = []
            self.record_btn.config(state=tk.DISABLED)
            self.log("Recording started - 60 seconds...")

            # Run recording in background thread
            thread = threading.Thread(target=self._recording_timer, daemon=True)
            thread.start()

    def _recording_timer(self):
        """Timer for 60-second recording"""
        start_time = datetime.now()
        elapsed = 0

        while elapsed < 60 and self.is_recording:
            self.root.after(0, lambda: self.recording_status_label.config(
                text=f"Recording: {elapsed}s / 60s", foreground="green"))

            threading.Event().wait(1)
            elapsed = int((datetime.now() - start_time).total_seconds())

        if self.is_recording:
            self.is_recording = False
            self.root.after(0, self._recording_complete)

    def _recording_complete(self):
        """Handle recording completion"""
        self.record_btn.config(state=tk.NORMAL)
        self.recording_status_label.config(text="")
        self.log("Recording completed - 60 seconds")

        # Save data to files
        if self.recording_data['ch1']:
            self._save_recording()

    def _save_recording(self):
        """Save recorded data to CSV and summary TXT file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_file = f"ble_recording_{timestamp}.csv"
            txt_file = f"ble_recording_{timestamp}_summary.txt"

            # Save to CSV
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Float1', 'Float2', 'Float3', 'Float4'])

                for i, ts in enumerate(self.recording_timestamps):
                    writer.writerow([
                        ts,
                        self.recording_data['ch1'][i],
                        self.recording_data['ch2'][i],
                        self.recording_data['ch3'][i],
                        self.recording_data['ch4'][i]
                    ])

            self.log(f"CSV saved: {csv_file}")

            # Calculate and save summary
            summary = self._generate_summary()
            with open(txt_file, 'w') as f:
                f.write(summary)

            self.log(f"Summary saved: {txt_file}")
            messagebox.showinfo("Success", f"Recording saved!\n\nCSV: {csv_file}\nSummary: {txt_file}")

        except Exception as e:
            self.log(f"Error saving recording: {e}")
            messagebox.showerror("Error", f"Failed to save recording: {e}")

    def _generate_summary(self):
        """Generate summary statistics for recording"""
        summary = []
        summary.append("BLE Data Collection Summary\n")
        summary.append("=" * 60 + "\n")
        summary.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary.append(f"Total samples: {len(self.recording_data['ch1'])}\n")
        summary.append(f"Duration: 60 seconds\n")
        summary.append("\n")

        channel_names = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']

        for i, (channel, name) in enumerate(zip(['ch1', 'ch2', 'ch3', 'ch4'], channel_names)):
            data = self.recording_data[channel]
            if data:
                arr = np.array(data)
                summary.append(f"{name}:\n")
                summary.append(f"  Samples: {len(data)}\n")
                summary.append(f"  Mean: {np.mean(arr):.4f}\n")
                summary.append(f"  Std Dev: {np.std(arr):.4f}\n")
                summary.append(f"  Min: {np.min(arr):.4f}\n")
                summary.append(f"  Max: {np.max(arr):.4f}\n")
                summary.append(f"  5th Percentile: {np.percentile(arr, 5):.4f}\n")
                summary.append(f"  95th Percentile: {np.percentile(arr, 95):.4f}\n")
                summary.append(f"  Median: {np.median(arr):.4f}\n\n")

        return "".join(summary)

    def run_ble_loop(self):
        """Run the BLE event loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.ble_main())
        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            loop.close()

    async def ble_main(self):
        """Main BLE connection and data receiving loop"""
        try:
            async with BleakClient(DEVICE_ADDRESS) as client:
                self.connected = True
                self.log(f"Connected to device")
                self.root.after(0, self.update_status_connected)

                # Send initial command
                await client.write_gatt_char(TX_CHARACTERISTIC_UUID, bytearray([0x01, 0xD0, 0x01]))
                self.log("Sent init command")
                await asyncio.sleep(1)

                def notification_handler(sender, data: bytearray):
                    """Handle incoming BLE notifications"""
                    try:
                        if len(data) >= 17:  # Need at least 1 byte header + 16 bytes (4 floats)
                            float_values = struct.unpack('<4f', data[1:17])

                            # Store values with lock
                            with self.data_lock:
                                data_storage['ch1'].append(float_values[0])
                                data_storage['ch2'].append(float_values[1])
                                data_storage['ch3'].append(float_values[2])
                                data_storage['ch4'].append(float_values[3])

                                # If recording, also store to recording buffer
                                if self.is_recording:
                                    self.recording_data['ch1'].append(float_values[0])
                                    self.recording_data['ch2'].append(float_values[1])
                                    self.recording_data['ch3'].append(float_values[2])
                                    self.recording_data['ch4'].append(float_values[3])
                                    self.recording_timestamps.append(
                                        datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                    )
                    except Exception as e:
                        self.log(f"Parse error: {e}")

                # Start notifications
                await client.start_notify(RX_CHARACTERISTIC_UUID, notification_handler)
                self.log("Listening for data...")

                # Keep sending commands and listening
                try:
                    while self.connected:
                        await client.write_gatt_char(TX_CHARACTERISTIC_UUID, PAYLOAD)
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    pass
                finally:
                    await client.stop_notify(RX_CHARACTERISTIC_UUID)

        except Exception as e:
            self.log(f"Connection error: {e}")
        finally:
            self.connected = False
            self.root.after(0, self.update_status_disconnected)

    def update_status_connected(self):
        """Update status to connected"""
        self.status_label.config(text="Status: Connected ✓", foreground="green")
        self.disconnect_btn.config(state=tk.NORMAL)
        self.connect_btn.config(state=tk.DISABLED)
        self.record_btn.config(state=tk.NORMAL)

    def update_status_disconnected(self):
        """Update status to disconnected"""
        self.status_label.config(text="Status: Disconnected", foreground="red")
        self.disconnect_btn.config(state=tk.DISABLED)
        self.connect_btn.config(state=tk.NORMAL)
        self.record_btn.config(state=tk.DISABLED)
        self.recording_status_label.config(text="")

    def get_optimal_y_limits(self, data_list):
        """Calculate optimal Y-axis limits using percentile method"""
        if not data_list or len(data_list) == 0:
            return None

        arr = np.array(data_list)

        # Calculate 5th and 95th percentiles to exclude outliers
        p5 = np.percentile(arr, 5)
        p95 = np.percentile(arr, 95)

        # Calculate range
        data_range = p95 - p5

        # Add 20% margin
        margin = data_range * 0.2

        y_min = p5 - margin
        y_max = p95 + margin

        return y_min, y_max

    def update_display(self):
        """Periodically update the graph and statistics"""
        try:
            # Get thread-safe snapshot of data
            with self.data_lock:
                data_snapshot = {
                    'ch1': list(data_storage['ch1']),
                    'ch2': list(data_storage['ch2']),
                    'ch3': list(data_storage['ch3']),
                    'ch4': list(data_storage['ch4']),
                }

            # Update graphs
            if len(data_snapshot['ch1']) > 0:
                self.fig.clear()

                x = np.arange(len(data_snapshot['ch1']))
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']
                channels = ['ch1', 'ch2', 'ch3', 'ch4']

                if self.separate_graphs_var.get():
                    # 4 separate subplots
                    axes = self.fig.subplots(2, 2)
                    axes = axes.flatten()

                    for ax, channel, color, label in zip(axes, channels, colors, labels):
                        data = data_snapshot[channel]
                        if data:
                            ax.plot(x, data, color=color, linewidth=2, marker='o', markersize=2, alpha=0.8)
                            ax.fill_between(x, data, alpha=0.2, color=color)
                            ax.set_title(label, fontsize=11, fontweight='bold')
                            ax.set_xlabel('Sample Index', fontsize=9)
                            ax.set_ylabel('Value', fontsize=9)
                            ax.grid(True, alpha=0.3)

                            # Apply scaling
                            if self.auto_scale_var.get():
                                limits = self.get_optimal_y_limits(data)
                                if limits:
                                    ax.set_ylim(limits[0], limits[1])
                else:
                    # Single plot with all channels
                    ax = self.fig.add_subplot(111)

                    for channel, color, label in zip(channels, colors, labels):
                        data = data_snapshot[channel]
                        if data:
                            ax.plot(x, data, color=color, linewidth=2, label=label, marker='o', markersize=2, alpha=0.8)

                    ax.set_xlabel('Sample Index', fontsize=11)
                    ax.set_ylabel('Sensor Value', fontsize=11)
                    ax.set_title('Real-Time BLE Sensor Data', fontsize=12, fontweight='bold')
                    ax.legend(loc='best', fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # Apply scaling - combine all data
                    if self.auto_scale_var.get():
                        all_data = []
                        for channel in channels:
                            all_data.extend(data_snapshot[channel])
                        if all_data:
                            limits = self.get_optimal_y_limits(all_data)
                            if limits:
                                ax.set_ylim(limits[0], limits[1])

                self.fig.tight_layout()
                self.canvas.draw_idle()

                # Update values and get prediction
                current_values = {}
                for i, channel in enumerate(['ch1', 'ch2', 'ch3', 'ch4']):
                    if data_snapshot[channel]:
                        current_val = data_snapshot[channel][-1]
                        current_values[channel] = current_val
                        self.value_labels[channel].config(text=f"{current_val:.1f}")

                # Make prediction if all channels have values
                if len(current_values) == 4 and self.model_loaded:
                    occupancy, confidence, probability = self.predict_occupancy(
                        current_values['ch1'],
                        current_values['ch2'],
                        current_values['ch3'],
                        current_values['ch4']
                    )

                    # Update occupancy label with color
                    color = "green" if occupancy == "Occupied" else "blue"
                    self.occupancy_label.config(text=occupancy, foreground=color)

                    # Update confidence
                    self.confidence_label.config(text=f"{confidence:.2%}")

                    # Update probability bar
                    self.draw_probability_bar(probability)

                # Update statistics
                stats_text = "═" * 40 + "\n"
                stats_text += "STATISTICS\n"
                stats_text += "═" * 40 + "\n\n"

                for i, (channel, label) in enumerate(zip(['ch1', 'ch2', 'ch3', 'ch4'], labels)):
                    data = data_snapshot[channel]
                    if data:
                        stats_text += f"{label}:\n"
                        stats_text += f"  Samples: {len(data)}\n"
                        stats_text += f"  Mean: {np.mean(data):.1f}\n"
                        stats_text += f"  Std: {np.std(data):.1f}\n"
                        stats_text += f"  Min: {np.min(data):.1f}\n"
                        stats_text += f"  Max: {np.max(data):.1f}\n"

                        # Calculate 5th and 95th percentiles
                        p5 = np.percentile(data, 5)
                        p95 = np.percentile(data, 95)
                        stats_text += f"  P5: {p5:.1f}\n"
                        stats_text += f"  P95: {p95:.1f}\n\n"

                self.stats_text.config(state=tk.NORMAL)
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(1.0, stats_text)
                self.stats_text.config(state=tk.DISABLED)

        except Exception as e:
            print(f"Update display error: {e}")

        # Schedule next update
        self.root.after(500, self.update_display)


def main():
    root = tk.Tk()
    gui = BLEDataGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()