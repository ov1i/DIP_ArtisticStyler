import PySimpleGUI as sg
from PIL import Image, ImageTk
import subprocess
import platform
import numpy as np
from src.color_matching.cm import match_colors
from src.conversion.color_space_conversions import *
from src.feature_fusion.edge_enhancement import edge_enhancement_wrapper
from src.feature_fusion.spectrum_extractor import feature_fusion_wrapper
from src.types.defines import global_vars

# Function to select a file
def select_file():
    if platform.system() == "Darwin":
        try:
            filepath = subprocess.check_output(
                '''osascript -e 'POSIX path of (choose file of type {"png", "jpg", "jpeg", "bmp"} with prompt "Select an image file")' ''',
                shell=True,
                text=True
            ).strip()
            return filepath
        except subprocess.CalledProcessError:
            return None
    else:
        return sg.popup_get_file(
            "Choose an image file",
            file_types=(("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"),),
            no_window=True
        )
    
def update_globals(values):
    global global_vars
    global_vars["kernel_size"] = int(values["-KERNEL_SIZE-"])
    global_vars["sigma"] = float(values["-SIGMA-"])
    global_vars["weight"] = float(values["-WEIGHT-"])
    global_vars["padding_flag"] = int(values["-PADDING_FLAG-"])

def default_values():
    global global_vars
    global_vars["kernel_size"] = 5
    global_vars["sigma"] = 1
    global_vars["weight"] = 0.6
    global_vars["padding_flag"] = 1


# Function to open and resize an image
def open_and_resize_image(filepath_or_pil_image, window, image_key, target_size=None):
    try:
        if isinstance(filepath_or_pil_image, str):
            img = Image.open(filepath_or_pil_image)
        else:
            img = filepath_or_pil_image

        # Resize dynamically
        if target_size is None:
            window_size = window.size
            target_size = (window_size[0] // 3, window_size[1] // 2)

        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        # Update GUI
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
        [
            sg.Button("Step-by-Step Algo", size=(20, 2), button_color=("black", "pink"), font=("Papyrus", 10, "bold")),
            sg.Button("Direct Result", size=(20, 2), button_color=("black", "pink"), font=("Papyrus", 10, "bold")),
            sg.Button("Exit", size=(20, 2), button_color=("black", "pink"), font=("Papyrus", 10, "bold")),
        ]
    ]
    return sg.Window("Home Menu", layout, background_color=BACKGROUND_COLOR, resizable=True, finalize=True)


# Function to create the Main Interface
def create_main_interface():
    BACKGROUND_COLOR = "#FFB6C1"
    layout = [
        [
            sg.Button("Open Initial Image", button_color=("black", "pink")),
            sg.Button("Open Style Image", button_color=("black", "pink")),
            sg.Push(background_color=BACKGROUND_COLOR),
            sg.Button("Back to Home Menu", button_color=("black", "pink")),
            
        ],
        [
            sg.Button("Color matching", button_color=("black", "pink")),
            sg.Button("Enhancing", button_color=("black", "pink")),
            sg.Button("Transfer style", button_color=("black", "pink")),
        ],
        [
            sg.Text("Kernel Size:", background_color=BACKGROUND_COLOR),
            sg.Slider(range=(1, 11), default_value=5, resolution=1, orientation='h', key="-KERNEL_SIZE-", background_color=BACKGROUND_COLOR),
        ],
         [
            sg.Text("Sigma:", background_color=BACKGROUND_COLOR),
            sg.Slider(range=(0.1, 5.0), default_value=1.0, resolution=0.1, orientation='h', key="-SIGMA-", background_color=BACKGROUND_COLOR),
        ],
        [
            sg.Text("Weight (w):", background_color=BACKGROUND_COLOR),
            sg.Slider(range=(0.1, 1.0), default_value=0.6, resolution=0.1, orientation='h', key="-WEIGHT-", background_color=BACKGROUND_COLOR),
        ],
         [
            sg.Text("Padding Flag:", background_color=BACKGROUND_COLOR),
            sg.Slider(range=(0, 1), default_value=1, resolution=1, orientation='h', key="-PADDING_FLAG-", background_color=BACKGROUND_COLOR),
        ],

        [
            sg.Column(
                [[sg.Image(key="-INITIAL_IMAGE-", background_color=BACKGROUND_COLOR, expand_x=True, expand_y=True)]],
                background_color=BACKGROUND_COLOR,
                expand_x=True,
                expand_y=True,
            ),
            sg.Column(
                [[sg.Image(key="-STYLE_IMAGE-", background_color=BACKGROUND_COLOR, expand_x=True, expand_y=True)]],
                background_color=BACKGROUND_COLOR,
                expand_x=True,
                expand_y=True,
            ),
            sg.Column(
                [[sg.Image(key="-RESULT_IMAGE-", background_color=BACKGROUND_COLOR, expand_x=True, expand_y=True)]],
                background_color=BACKGROUND_COLOR,
                expand_x=True,
                expand_y=True,
            ),
        ],
    ]
    return sg.Window("Image Style Transfer", layout, background_color=BACKGROUND_COLOR, resizable=True, size=(1500, 900), finalize=True)

def create_direct_interface():
    BACKGROUND_COLOR = "#FFB6C1"
    layout = [
        [
            sg.Button("Open Initial Image", button_color=("black", "pink")),
            sg.Button("Open Style Image", button_color=("black", "pink")),
            sg.Push(background_color=BACKGROUND_COLOR),
            sg.Button("Back to Home Menu", button_color=("black", "pink")),
            
        ],
        [
        
            sg.Button("Transfer style", button_color=("black", "pink")),
        ],
        [
            sg.Column(
                [[sg.Image(key="-INITIAL_IMAGE-", background_color=BACKGROUND_COLOR, expand_x=True, expand_y=True)]],
                background_color=BACKGROUND_COLOR,
                expand_x=True,
                expand_y=True,
            ),
            sg.Column(
                [[sg.Image(key="-STYLE_IMAGE-", background_color=BACKGROUND_COLOR, expand_x=True, expand_y=True)]],
                background_color=BACKGROUND_COLOR,
                expand_x=True,
                expand_y=True,
            ),
            sg.Column(
                [[sg.Image(key="-RESULT_IMAGE-", background_color=BACKGROUND_COLOR, expand_x=True, expand_y=True)]],
                background_color=BACKGROUND_COLOR,
                expand_x=True,
                expand_y=True,
            ),
        ],
    ]
    return sg.Window("Image Style Transfer", layout, background_color=BACKGROUND_COLOR, resizable=True, size=(1500, 900), finalize=True)



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
            method = 0
            window = create_main_interface()

        elif event == "Back to Home Menu":
            window.close()
            window = create_home_menu()

        elif event == "Direct Result":
            window.close()
            method = 1
            window = create_direct_interface()
            
        elif event == "Open Initial Image":
            filepath = select_file()
            if filepath:
                initial_image = open_and_resize_image(filepath, window, "-INITIAL_IMAGE-")

        elif event == "Open Style Image":
            filepath = select_file()
            if filepath:
                style_image = open_and_resize_image(filepath, window, "-STYLE_IMAGE-")

        elif event == "Color matching":
            if not initial_image or not style_image:
                sg.popup_error("Please load both images before color.")
                continue
            try:
                update_globals(values)
                initial_array = np.array(initial_image)
                style_array = np.array(style_image)
                initial_lab = bgr_rgb_to_lab(initial_array)
                style_lab = bgr_rgb_to_lab(style_array)
                matched_lab = match_colors(initial_lab, style_lab)
                matched_bgr = lab_to_bgr_rgb(matched_lab, 1)
                result_pil = Image.fromarray(matched_bgr.astype("uint8"))
                open_and_resize_image(result_pil, window, "-RESULT_IMAGE-")
            except Exception as e:
                sg.popup_error(f"Error during color matching: {e}")

        elif event == "Enhancing":
            if not initial_image or not style_image:
                sg.popup_error("Please load both images before color.")
                continue
            try:
                update_globals(values)
                initial_array = np.array(initial_image)
                style_array = np.array(style_image)
                initial_lab = bgr_rgb_to_lab(initial_array)
                style_lab = bgr_rgb_to_lab(style_array)
                matched_lab = match_colors(initial_lab, style_lab)
                matched_bgr = lab_to_bgr_rgb(matched_lab, 1)
                result_image = edge_enhancement_wrapper(matched_bgr)
                result_pil = Image.fromarray(result_image.astype("uint8"))
                open_and_resize_image(result_pil, window, "-RESULT_IMAGE-")
            except Exception as e:
                sg.popup_error(f"Error during enhancing: {e}")
        elif event == "Transfer style":
            if not initial_image or not style_image:
                sg.popup_error("Please load both images before color.")
                continue
            try:
                if method == 0:
                    update_globals(values)
                else:
                    default_values()
                initial_array = np.array(initial_image)
                style_array = np.array(style_image)
                initial_lab = bgr_rgb_to_lab(initial_array)
                style_lab = bgr_rgb_to_lab(style_array)
                matched_lab = match_colors(initial_lab, style_lab)
                matched_bgr = lab_to_bgr_rgb(matched_lab, 1)
                enh_image = edge_enhancement_wrapper(matched_bgr)
                enh_image_lab = bgr_rgb_to_lab(enh_image)
                result_image_lab = feature_fusion_wrapper(enh_image_lab, style_lab)
                result_image = lab_to_bgr_rgb(result_image_lab, 1)
                result_pil = Image.fromarray(result_image.astype("uint8"))
                open_and_resize_image(result_pil, window, "-RESULT_IMAGE-")
                
            except Exception as e:
                sg.popup_error(f"Error during style transfer: {e}")

        elif event == "Reset Images":
            initial_image = None
            style_image = None
            window["-INITIAL_IMAGE-"].update(data=None)
            window["-STYLE_IMAGE-"].update(data=None)
            window["-RESULT_IMAGE-"].update(data=None)

        elif event == "Settings":
            sg.popup("Settings feature is under development.")

    window.close()
    