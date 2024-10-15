import sys
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # Use the Qt5 backend for matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QLabel, QMessageBox, QHBoxLayout, QSlider
)
from PyQt5.QtCore import Qt
import numpy as np

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        # Create a figure and two y-axes
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()
        super(MplCanvas, self).__init__(self.fig)

class BatchTrimmer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Batch Trimmer with Scatter Plots (Index X-axis)")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.data = None
        self.batches = []
        self.current_batch_idx = 0
        self.trimmed_data = []
        self.discarded_batches = set()
        self.trim_start_idx = None
        self.trim_end_idx = None

        # Set up UI
        self.initUI()

    def initUI(self):
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Load CSV button
        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv)
        self.layout.addWidget(self.load_btn)

        # Label for batch information
        self.batch_label = QLabel("No batch loaded.")
        self.layout.addWidget(self.batch_label)

        # Status label for user instructions
        self.status_label = QLabel("Please load a CSV file to begin.")
        self.layout.addWidget(self.status_label)

        # Matplotlib Canvas for plotting
        self.canvas = MplCanvas(self, width=10, height=6, dpi=100)
        self.layout.addWidget(self.canvas)

        # Sliders Layout
        sliders_layout = QHBoxLayout()

        # Start Slider
        self.slider_start_label = QLabel("Start Row:")
        sliders_layout.addWidget(self.slider_start_label)

        self.slider_start = QSlider(Qt.Horizontal)
        self.slider_start.setMinimum(1)
        self.slider_start.setEnabled(False)
        self.slider_start.valueChanged.connect(self.update_trim)
        sliders_layout.addWidget(self.slider_start)

        # End Slider
        self.slider_end_label = QLabel("End Row:")
        sliders_layout.addWidget(self.slider_end_label)

        self.slider_end = QSlider(Qt.Horizontal)
        self.slider_end.setMinimum(1)
        self.slider_end.setEnabled(False)
        self.slider_end.valueChanged.connect(self.update_trim)
        sliders_layout.addWidget(self.slider_end)

        self.layout.addLayout(sliders_layout)

        # Navigation buttons layout
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("Previous Batch")
        self.prev_btn.clicked.connect(self.prev_batch)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next Batch")
        self.next_btn.clicked.connect(self.next_batch)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)

        # Discard Batch button
        self.discard_btn = QPushButton("Discard Batch")
        self.discard_btn.clicked.connect(self.discard_batch)
        self.discard_btn.setEnabled(False)
        nav_layout.addWidget(self.discard_btn)

        self.layout.addLayout(nav_layout)

        # Save button
        self.save_btn = QPushButton("Save Retained Data")
        self.save_btn.clicked.connect(self.save_csv)
        self.save_btn.setEnabled(False)
        self.layout.addWidget(self.save_btn)

    def load_csv(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            try:
                self.data = pd.read_csv(fileName)
                # Normalize column names to lowercase
                self.data.columns = self.data.columns.str.lower()
                required_cols = {'batch', 'data_point', 'value'}
                if not required_cols.issubset(self.data.columns):
                    QMessageBox.critical(self, "Error", f"CSV must contain columns: {required_cols}")
                    return

                # Assign row_number globally for each batch
                self.data['row_number'] = self.data.groupby('batch').cumcount() + 1

                # Sort data by batch and row_number
                self.data.sort_values(by=['batch', 'row_number'], inplace=True)
                self.data.reset_index(drop=True, inplace=True)

                self.batches = sorted(self.data['batch'].unique())
                self.current_batch_idx = 0
                self.trimmed_data = []
                self.discarded_batches = set()
                self.trim_start_idx = None
                self.trim_end_idx = None

                self.save_btn.setEnabled(False)
                self.discard_btn.setEnabled(True)
                self.prev_btn.setEnabled(False)
                self.next_btn.setEnabled(len(self.batches) > 1)

                self.display_batch()
                self.status_label.setText("Use the sliders to select the data range to retain.")

                # Debugging: Print the first few rows of the loaded data
                print("Loaded Data:")
                print(self.data.head())

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")

    def display_batch(self):
        if not self.batches:
            return

        # Get current batch
        batch = self.batches[self.current_batch_idx]
        if batch in self.discarded_batches:
            QMessageBox.information(self, "Info", f"Batch '{batch}' has been discarded.")
            self.next_batch()
            return

        batch_data = self.data[self.data['batch'] == batch].copy().reset_index(drop=True)
        self.current_batch_data = batch_data

        self.batch_label.setText(f"Batch {self.current_batch_idx + 1} of {len(self.batches)}: {batch}")

        # Debugging: Print the first few rows of the current batch
        print(f"\nDisplaying Batch: {batch}")
        print(batch_data.head())

        # Clear previous plots
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.fig.subplots_adjust(right=0.75)  # Adjust to prevent overlap

        # Plot data_point as scatter
        x = batch_data['row_number']
        self.canvas.ax1.scatter(x, batch_data['data_point'], color='blue', label='Data Point', s=10)
        self.canvas.ax1.set_xlabel('Row Number')
        self.canvas.ax1.set_ylabel('Data Point', color='blue')
        self.canvas.ax1.tick_params(axis='y', labelcolor='blue')

        # Plot value as scatter
        self.canvas.ax2.scatter(x, batch_data['value'], color='red', label='Value', s=10)
        self.canvas.ax2.set_ylabel('Value', color='red')
        self.canvas.ax2.tick_params(axis='y', labelcolor='red')

        # Format x-axis with row numbers
        self.canvas.ax1.set_xticks(np.arange(1, len(batch_data) + 1, max(1, len(batch_data)//10)))
        self.canvas.ax1.set_xticklabels([str(i) for i in self.canvas.ax1.get_xticks()], rotation=45)

        # Highlight retained region if already selected
        if self.trim_start_idx is not None and self.trim_end_idx is not None:
            start_idx = self.trim_start_idx
            end_idx = self.trim_end_idx
            self.canvas.ax1.axvspan(start_idx, end_idx, color='green', alpha=0.3, label='Retained Region')
            # Add vertical lines for start and end
            self.canvas.ax1.axvline(start_idx, color='green', linestyle='--')
            self.canvas.ax1.axvline(end_idx, color='green', linestyle='--')

        self.canvas.draw()

        # Initialize sliders
        num_points = len(batch_data)
        if num_points == 0:
            QMessageBox.warning(self, "Warning", f"Batch '{batch}' has no data.")
            return

        self.slider_start.setEnabled(True)
        self.slider_end.setEnabled(True)

        self.slider_start.setMinimum(1)
        self.slider_start.setMaximum(num_points)
        self.slider_start.setValue(1)
        self.slider_start.setTickInterval(1)
        self.slider_start.setSingleStep(1)

        self.slider_end.setMinimum(1)
        self.slider_end.setMaximum(num_points)
        self.slider_end.setValue(num_points)
        self.slider_end.setTickInterval(1)
        self.slider_end.setSingleStep(1)

        self.trim_start_idx = 1
        self.trim_end_idx = num_points

    def update_trim(self):
        if not self.batches:
            return

        batch = self.batches[self.current_batch_idx]
        if batch in self.discarded_batches:
            return

        start = self.slider_start.value()
        end = self.slider_end.value()

        # Ensure start <= end
        if start > end:
            if self.sender() == self.slider_start:
                self.slider_end.setValue(start)
                end = start
            else:
                self.slider_start.setValue(end)
                start = end

        self.trim_start_idx = start
        self.trim_end_idx = end

        batch_data = self.current_batch_data

        # Debugging: Print the selected range
        print(f"\nSelected range for Batch '{batch}': Row {start} to Row {end}")

        # Update the plot
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.fig.subplots_adjust(right=0.75)  # Adjust to prevent overlap

        # Plot data_point as scatter
        x = batch_data['row_number']
        self.canvas.ax1.scatter(x, batch_data['data_point'], color='blue', label='Data Point', s=10)
        self.canvas.ax1.set_xlabel('Row Number')
        self.canvas.ax1.set_ylabel('Data Point', color='blue')
        self.canvas.ax1.tick_params(axis='y', labelcolor='blue')

        # Plot value as scatter
        self.canvas.ax2.scatter(x, batch_data['value'], color='red', label='Value', s=10)
        self.canvas.ax2.set_ylabel('Value', color='red')
        self.canvas.ax2.tick_params(axis='y', labelcolor='red')

        # Format x-axis with row numbers
        self.canvas.ax1.set_xticks(np.arange(1, len(batch_data) + 1, max(1, len(batch_data)//10)))
        self.canvas.ax1.set_xticklabels([str(i) for i in self.canvas.ax1.get_xticks()], rotation=45)

        # Highlight retained region
        self.canvas.ax1.axvspan(start, end, color='green', alpha=0.3, label='Retained Region')
        # Add vertical lines for start and end
        self.canvas.ax1.axvline(start, color='green', linestyle='--')
        self.canvas.ax1.axvline(end, color='green', linestyle='--')

        self.canvas.draw()

        # Update status label
        self.status_label.setText(f"Selected range: Row {start} to Row {end}")

    def prev_batch(self):
        if self.current_batch_idx > 0:
            self.current_batch_idx -= 1
            self.trim_start_idx = 1
            self.trim_end_idx = len(self.current_batch_data)
            self.display_batch()
            self.update_nav_buttons()

    def next_batch(self):
        if self.current_batch_idx < len(self.batches) - 1:
            # Save current trimmed data
            self.save_current_trim()
            self.current_batch_idx += 1
            self.display_batch()
            self.update_nav_buttons()
            self.save_btn.setEnabled(True)
        else:
            QMessageBox.information(self, "Info", "You have reached the last batch.")
            self.update_nav_buttons()

    def discard_batch(self):
        if not self.batches:
            return

        batch = self.batches[self.current_batch_idx]
        if batch in self.discarded_batches:
            QMessageBox.information(self, "Info", f"Batch '{batch}' is already discarded.")
            return

        reply = QMessageBox.question(
            self, 'Discard Batch',
            f"Are you sure you want to discard Batch '{batch}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.discarded_batches.add(batch)
            QMessageBox.information(self, "Discarded", f"Batch '{batch}' has been discarded.")
            self.status_label.setText(f"Batch '{batch}' discarded. Moving to next batch.")
            self.next_batch()

    def save_current_trim(self):
        if not self.batches:
            return

        batch = self.batches[self.current_batch_idx]
        if batch in self.discarded_batches:
            return

        batch_data = self.current_batch_data.copy()
        trimmed = batch_data.iloc[self.trim_start_idx - 1:self.trim_end_idx]  # Adjusting for 1-based index
        self.trimmed_data.append(trimmed)

        # Debugging: Print the trimmed data being saved
        print(f"\nSaving trimmed data for Batch '{batch}':")
        print(trimmed.head())

    def save_csv(self):
        # Save the current batch's trimmed data
        if self.batches:
            self.save_current_trim()

        if not self.trimmed_data and not self.discarded_batches:
            QMessageBox.warning(self, "No Data", "No trimmed data to save.")
            return

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            try:
                # Concatenate all trimmed data
                if self.trimmed_data:
                    final_df = pd.concat(self.trimmed_data).reset_index(drop=True)
                else:
                    final_df = pd.DataFrame(columns=self.data.columns)

                final_df.to_csv(fileName, index=False)
                QMessageBox.information(self, "Success", f"Trimmed data saved to '{fileName}'.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save CSV: {e}")

    def update_nav_buttons(self):
        self.prev_btn.setEnabled(self.current_batch_idx > 0)
        self.next_btn.setEnabled(self.current_batch_idx < len(self.batches) - 1)

def main():
    app = QApplication(sys.argv)
    window = BatchTrimmer()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
