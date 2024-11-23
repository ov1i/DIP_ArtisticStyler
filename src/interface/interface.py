import PySimpleGUI as sg
from PIL import Image, ImageTk
import subprocess
import platform
import numpy as np
from src.color_matching.cm import match_colors
from src.conversion.color_space_conversions import *
from src.feature_fusion.edge_enhancement import edge_enhancement_wrapper


# Function to select a file
def select_file():
    """Use an external file picker on macOS or PySimpleGUI file picker."""
    if platform.system() == "Darwin":  # macOS
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


# Function to open and resize an image
def open_and_resize_image(filepath_or_pil_image, window, image_key, target_size=(640, 480)):
    """
    Open an image (from filepath or PIL.Image), resize it, and display it in the GUI.
    """
    try:
        if isinstance(filepath_or_pil_image, str):
            img = Image.open(filepath_or_pil_image)
        else:
            img = filepath_or_pil_image  # Assume input is already a PIL Image object

        # Resize image
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        # Update the GUI
        window[image_key].update(data=img_tk)
        window[image_key].image = img_tk  # Prevent garbage collection

        return img_resized
    except Exception as e:
        sg.popup_error(f"Error loading image: {e}")
        return None


# Function to create the Home Menu
def create_home_menu():
    BACKGROUND_COLOR = "#FFB6C1"
    layout = [
        [sg.Text("Welcome to the Home Menu", font=("Papyrus", 30), justification="center", background_color=BACKGROUND_COLOR)],
        [sg.Button("Step-by-Step Algo", size=(20, 2), button_color=("black", "pink"), font=("Papyrus", 10, "bold")),
         sg.Button("Direct Result", size=(20, 2), button_color=("black", "pink"), font=("Papyrus", 10, "bold")),
         sg.Button("Exit", size=(20, 2), button_color=("black", "pink"), font=("Papyrus", 10, "bold"))]
    ]
    return sg.Window("Home Menu", layout, background_color=BACKGROUND_COLOR, size=(600, 300), finalize=True)


# Function to create the Main Interface
def create_main_interface():
    BACKGROUND_COLOR = "#FFB6C1"
    layout = [
        [
            sg.Button("Open Initial Image", button_color=("black", "pink")),
            sg.Button("Open Style Image", button_color=("black", "pink")),
            sg.Button("Back to Home Menu", button_color=("black", "pink")),
            sg.Button("Exit", button_color=("black", "pink")),
        ],
        [
            sg.Column(
                [[sg.Image(key="-INITIAL_IMAGE-", size=(300, 300), background_color=BACKGROUND_COLOR)]],
                background_color=BACKGROUND_COLOR,
                size=(500, 500),
            ),
            sg.Column(
                [[sg.Image(key="-STYLE_IMAGE-", size=(300, 300), background_color=BACKGROUND_COLOR)]],
                background_color=BACKGROUND_COLOR,
                size=(500, 500),
            ),
            sg.Column(
                [[sg.Image(key="-RESULT_IMAGE-", size=(300, 300), background_color=BACKGROUND_COLOR)]],
                background_color=BACKGROUND_COLOR,
                size=(500, 500),
            ),
            sg.Column(
                [
                    [sg.Button("Transfer Style", button_color=("black", "pink"))],
                ],
                background_color=BACKGROUND_COLOR,
                size=(150, 500),
                vertical_alignment="center",
                element_justification="center",
            ),
        ],
    ]
    return sg.Window("Image Style Transfer", layout, background_color=BACKGROUND_COLOR, size=(1700, 600), finalize=True)


# Main Function to Run the GUI
def main():
    window = create_home_menu()
    initial_image = None
    style_image = None

    while True:
        event, values = window.read(timeout=10)
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Step-by-Step Algo":
            window.close()
            window = create_main_interface()

        elif event == "Back to Home Menu":
            window.close()
            window = create_home_menu()

        elif event == "Open Initial Image":
            filepath = select_file()
            if filepath:
                initial_image = open_and_resize_image(filepath, window, "-INITIAL_IMAGE-")
                if initial_image:
                    print(f"Loaded Initial Image with size: {initial_image.size}")

        elif event == "Open Style Image":
            filepath = select_file()
            if filepath:
                style_image = open_and_resize_image(filepath, window, "-STYLE_IMAGE-")
                if style_image:
                    print(f"Loaded Style Image with size: {style_image.size}")

        elif event == "Transfer Style":
            if not initial_image or not style_image:
                sg.popup_error("Please load both images before transferring style.")
                continue

            try:
                # Convert images to NumPy arrays
                initial_array = np.array(initial_image)
                style_array = np.array(style_image)

                # Perform color matching
                initial_lab = bgr_rgb_to_lab(initial_array)
                style_lab = bgr_rgb_to_lab(style_array)
                matched_lab = match_colors(initial_lab, style_lab)
                matched_bgr = lab_to_bgr_rgb(matched_lab, 1)

                # Perform edge enhancement
                result_image = edge_enhancement_wrapper(matched_bgr)

                # Convert back to PIL Image
                result_pil = Image.fromarray(result_image.astype("uint8"))

                open_and_resize_image(result_pil, window, "-RESULT_IMAGE-")

                print("Style transfer completed.")
            except Exception as e:
                sg.popup_error(f"Error during style transfer: {e}")

    window.close()


