import PySimpleGUI as sg
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import time

# Initialize tkinter root for file dialogs (required for macOS)
root = tk.Tk()
root.withdraw()  # Hide the tkinter root window

# Set macOS-specific options to use ttk buttons for better compatibility
sg.set_options(use_ttk_buttons=True)

def select_file():
    """Use Tkinter's file dialog to select an image file."""
    filename = filedialog.askopenfilename(
        title="Choose an image file",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    return filename

def open_image(filename, window):
    """Open an image and update the window to display it."""
    try:
        if filename:
            # Open and resize image to fit within 400x400 pixels (adjust size as needed)
            img = Image.open(filename)
            img.thumbnail((400, 400))
            
            # Convert image to PhotoImage format for PySimpleGUI
            img_tk = ImageTk.PhotoImage(img)
            
            # Update the image element with the loaded image
            window['-IMAGE-'].update(data=img_tk)
            
            # Store the image reference to prevent garbage collection (necessary for macOS)
            window['-IMAGE-'].ImageData = img_tk
        else:
            print("No file selected.")
    except Exception as e:
        sg.popup_error(f"An error occurred: {e}")

# Define the layout of the window with a button to open images and an image display area
layout = [
    [sg.Button("Open Image")],
    [sg.Image(key="-IMAGE-")],  # Placeholder for the image display
]

# Create the window
window = sg.Window("Image Viewer", layout, finalize=True)

# Event loop to handle the "Open Image" button click
while True:
    # Read events with a small timeout to allow for smoother GUI updates on macOS
    event, values = window.read(timeout=10)
    
    if event == sg.WIN_CLOSED:
        break
    elif event == "Open Image":
        # Call the custom file dialog function
        filename = select_file()
        
        # Only attempt to open the image if a valid filename is returned
        if filename:
            open_image(filename, window)
        else:
            print("No file selected by the user.")

    # Small delay to avoid potential crashes on macOS
    time.sleep(0.01)

# Close the window when done
window.close()