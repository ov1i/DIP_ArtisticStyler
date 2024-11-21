import PySimpleGUI as sg
from PIL import Image, ImageTk
import subprocess
import time
import platform
import sys
sys.path.insert(1, 'src/color_matching')
sys.path.insert(1, 'src/conversion' )
sys.path.insert(1, 'src/feature_fusion' )

import conversion.color_space_conversions as cc
import color_matching.cm as cm
from feature_fusion.edge_enhancement import edge_enhancement_wrapper as enh_proc

def select_file():
    """Use an external file picker on macOS to select an image file."""
    if platform.system() == 'Darwin':  # macOS
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
        return sg.popup_get_file(
            "Choose an image file", 
            file_types=(("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"),),
            no_window=True
        )

def open_and_resize_image(filename, window, image_key, target_size=(640, 480)):
    """Open an image, resize it to target dimensions, and update the window to display it."""
    try:
        if filename:
            # Open the image
            img = Image.open(filename)
            if img.size == target_size:
                print(f"The image already has the target dimensions: {target_size}.")
            else:
                print(f"Original image size: {img.size}. Resizing to {target_size}.")
            
            # Resize the image to the target size
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert resized image to PhotoImage format for PySimpleGUI
            img_tk = ImageTk.PhotoImage(img_resized)
            
            # Update the image element with the resized image
            window[image_key].update(data=img_tk)
            
            # Keep a reference to prevent garbage collection
            window[image_key].image = img_tk
            
            # Return the resized image for further computation
            return img_resized
        else:
            print("No file selected.")
    except Exception as e:
        sg.popup_error(f"An error occurred: {e}")
        return None

# Define a background color for the window


def interface_wrapper():
    BACKGROUND_COLOR = "#FFB6C1"  # Dark gray background

# Define the layout of the window with a consistent background color
    layout = [
    [sg.Button("Open Initial Image", button_color=("black", "pink")), 
     sg.Button("Open the Image Whose Style to Be Transferred", button_color=("black", "pink")), 
     sg.Button("Transfer the style", button_color=("black", "pink")), 
     sg.Button("Exit", button_color=("black", "pink"))],
     
    [
        sg.Column(
            [
                [sg.Image(key="-INITIAL_IMAGE-", size=(500, 500))]  # Placeholder for Initial Image
            ],
            element_justification="center",
            background_color=BACKGROUND_COLOR,  # Match window background
            size=(500, 500),
        ),
        sg.Column(
            [
                [sg.Image(key="-STYLE_IMAGE-", size=(500, 500))]  # Placeholder for Style Image
            ],
            element_justification="center",
            background_color=BACKGROUND_COLOR,  # Match window background
            size=(500, 500),
        )
    ]
    ]

# Create the window with a fixed background color
    window = sg.Window(
    "Image Viewer", 
    layout, 
    finalize=True, 
    background_color=BACKGROUND_COLOR,  # Set background color for the window
    size=(1000, 600)  # Optional: Explicit window size
    )

# Event loop to handle button clicks
    while True:
     event, values = window.read(timeout=10)
     if event == sg.WIN_CLOSED:
        break
     elif event == "Open Initial Image":
        filename = select_file()
        if filename:
            resized_img = open_and_resize_image(filename, window, "-INITIAL_IMAGE-")
            # resized_img can now be used for further computations
            if resized_img:
                print(f"Resized Initial Image Size: {resized_img.size}")
     elif event == "Open the Image Whose Style to Be Transferred":
        filename = select_file()
        if filename:
            resized_img1 = open_and_resize_image(filename, window, "-STYLE_IMAGE-")
            # resized_img can now be used for further computations
            if resized_img1:
                print(f"Resized Style Image Size: {resized_img.size}")
     elif event =="Transfer the style"  :
        initial_img_lab = cc.bgr_rgb_to_lab(resized_img)
        painting_img_lab = cc.bgr_rgb_to_lab(resized_img1)
        cm_img_lab = cm.match_colors(initial_img_lab, painting_img_lab)
        cm_img_bgr = cc.lab_to_bgr_rgb(cm_img_lab, 1)
        enhanced_img=enh_proc(cm_img_bgr)
        img = open_and_resize_image(enhanced_img, window, "-STYLE_IMAGE-")

     elif event == "Exit":
        break

    time.sleep(0.01)

# Close the window when done
    window.close()
interface_wrapper()    