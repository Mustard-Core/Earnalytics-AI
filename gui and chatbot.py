import customtkinter as ctk






        
# ------------------------------
#  AI Salary Predictor GUI (Frontend Only)
# ------------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Create main window
app = ctk.CTk()
app.title("AI Salary Predictor")

app.state("zoomed")  # Open fullscreen
app.geometry("1300x1500")
app.configure(fg_color="#101820")

#  HEADER
header_frame = ctk.CTkFrame(app, fg_color="#1A1A2E", corner_radius=0)
header_frame.pack(fill="x")

header_label = ctk.CTkLabel(
    header_frame,
    text="💼  Earnlytics: AI Salary Predictor",
    font=("Arial Rounded MT Bold", 28),
    text_color="white"
)
header_label.pack(padx=20, pady=20)

# ------------------------------
#  MAIN AREA (Scrollable)
# ------------------------------
main_container = ctk.CTkScrollableFrame(app, fg_color="#16213E", corner_radius=20)
main_container.pack(fill="both", expand=True, padx=40, pady=(20, 0))

# Left Side – Input Fields
input_frame = ctk.CTkFrame(main_container, fg_color="#0F3460", corner_radius=15)
input_frame.pack(side="left", fill="y", expand=True, padx=20, pady=20)



entries = {}


# Predict Button (Dummy)
predict_button = ctk.CTkButton(
    input_frame,
    text="Predict Salary",
    height=40,
    width=200,
    font=("Arial Rounded MT Bold", 16),
    fg_color="#533483",
    hover_color="#6D44A0"
)
predict_button.pack(pady=25)





# ------------------------------
#  CHATBOT SECTION
# ------------------------------
chat_frame = ctk.CTkFrame(app, fg_color="#1A1A2E", corner_radius=0)
chat_frame.pack(fill="x", side="bottom")

ctk.CTkLabel(
    chat_frame,
    text="🤖 Chat with Earnlytics Assistant",
    font=("Arial Bold", 18),
    text_color="white"
).pack(anchor="w", padx=20, pady=(10, 0))

chat_display = ctk.CTkTextbox(chat_frame, height=140, font=("Consolas", 13))
chat_display.pack(fill="x", padx=20, pady=10)

# Auto greeting
chat_display.insert("end", "Earnlytics :Hello! I’m here to help you navigate the AI Salary Predictor.\n")
chat_display.insert("end", "Feel free to ask me questions about the system or salary insights.\n\n")

chat_entry = ctk.CTkEntry(chat_frame, placeholder_text="Type your message here...")


chat_entry.pack(side="left", fill="x", expand=True, padx=(20, 10), pady=(0, 20))

send_button = ctk.CTkButton(
    chat_frame,
    text="Send",
    width=100,
    height=35,
    fg_color="#533483",
    hover_color="#6D44A0",
    font=("Arial Rounded MT Bold", 14)
)


send_button.pack(side="right", padx=(0, 20), pady=(0, 20))




        
# ------------------------------
#  RUN APP
# ------------------------------
app.mainloop()



