import PySimpleGUI as sg
from src.color_matching.cm import match_colors
from src.conversion.color_space_conversions import *
from src.feature_fusion.edge_enhancement import edge_enhancement_wrapper

from src.interface import interface_wrapper
# Define the layout of the home menu
def home_menu():
    BACKGROUND_COLOR = "#FFB6C1"
    
    # Layout with centered text and buttons
    layout = [
        [sg.Push(background_color=BACKGROUND_COLOR), sg.Text("Welcome to the Home Menu", font=("Papyrus", 30), justification="center", background_color=BACKGROUND_COLOR), sg.Push(background_color=BACKGROUND_COLOR)],
        [sg.Push(background_color=BACKGROUND_COLOR), sg.Button("Step-by-step algo", size=(20, 2), button_color=("black", "pink"), font=("Papyrus", 10, "bold")), sg.Button("Direct result", size=(20, 2), button_color=("black", "pink"), font=("Papyrus", 10, "bold")), sg.Button("Exit", size=(20, 2), button_color=("black", "pink"), font=("Papyrus", 10, "bold")), sg.Push(background_color=BACKGROUND_COLOR)]
    ]
    
    window = sg.Window("Home Menu", layout, size=(700, 200), background_color=BACKGROUND_COLOR, finalize=True)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "Exit":
            break

        elif event == "Step-by-step algo":
            window.close()
            interface_wrapper()

        elif event == "Direct result":
            sg.popup("You selected Direct result")
    
    window.close()
