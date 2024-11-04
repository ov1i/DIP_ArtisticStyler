import PySimpleGUI as sg
from PIL import Image, ImageTk
import subprocess
import time
import platform

def select_file():
    """Use an external file picker on macOS to select an image file."""
    if platform.system() == 'Darwin':  # macOS
        # Use an AppleScript-based file dialog, which avoids Tkinter issues on macOS
        try:
            filepath = subprocess.check_output(
                '''osascript -e 'POSIX path of (choose file of type {"png", "jpg", "jpeg", "bmp"} with prompt "Select an image file")' ''',
                shell=True,
                text=True
            ).strip()
            return filepath
        except subprocess.CalledProcessError:
            return None  # Return None if the user cancels
    else:
        # Use PySimpleGUI's file dialog on non-macOS platforms
        return sg.popup_get_file(
            "Choose an image file", 
            file_types=(("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"),),
            no_window=True
        )

def open_image(filename, window):
    """Open an image and update the window to display it."""
    try:
        if filename:
            # Open and resize image to fit within 400x400 pixels (adjust as needed)
            img = Image.open(filename)
            img.thumbnail((400, 400))
            
            # Convert image to PhotoImage format for PySimpleGUI
            img_tk = ImageTk.PhotoImage(img)
            
            # Update the image element with the loaded image
            window['-IMAGE-'].update(data=img_tk)
            
            # Keep a reference to prevent garbage collection
            window['-IMAGE-'].image = img_tk
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
    # Read events with a small timeout to allow smoother GUI updates on macOS
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