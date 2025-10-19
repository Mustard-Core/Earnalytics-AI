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

ctk.CTkLabel(input_frame, text="Employee Information", font=("Arial Bold", 22)).pack(pady=15)

entries = {}

# --- Age ---
frame = ctk.CTkFrame(input_frame, fg_color="#16213E", corner_radius=10)
frame.pack(fill="x", pady=8, padx=15)
ctk.CTkLabel(frame, text="Age", font=("Arial", 16)).pack(side="left", padx=10, pady=10)
entries["Age"] = ctk.CTkEntry(frame, placeholder_text="Enter age")
entries["Age"].pack(side="right", fill="x", expand=True, padx=10, pady=10)

# --- Gender Dropdown ---
frame = ctk.CTkFrame(input_frame, fg_color="#16213E", corner_radius=10)
frame.pack(fill="x", pady=8, padx=15)
ctk.CTkLabel(frame, text="Gender", font=("Arial", 16)).pack(side="left", padx=10, pady=10)
entries["Gender"] = ctk.CTkOptionMenu(frame, values=["Male", "Female", "Other"])
entries["Gender"].pack(side="right", fill="x", expand=True, padx=10, pady=10)

# --- Education Dropdown ---
frame = ctk.CTkFrame(input_frame, fg_color="#16213E", corner_radius=10)
frame.pack(fill="x", pady=8, padx=15)
ctk.CTkLabel(frame, text="Education Level", font=("Arial", 16)).pack(side="left", padx=10, pady=10)
entries["Education Level"] = ctk.CTkOptionMenu(
    frame, values=["High School", "Diploma", "Bachelor's", "Master's", "PhD"]
)
entries["Education Level"].pack(side="right", fill="x", expand=True, padx=10, pady=10)

# --- Experience ---
frame = ctk.CTkFrame(input_frame, fg_color="#16213E", corner_radius=10)
frame.pack(fill="x", pady=8, padx=15)
ctk.CTkLabel(frame, text="Years of Experience", font=("Arial", 16)).pack(side="left", padx=10, pady=10)
entries["Experience"] = ctk.CTkEntry(frame, placeholder_text="Enter years of experience")
entries["Experience"].pack(side="right", fill="x", expand=True, padx=10, pady=10)

# --- Job Title ---
frame = ctk.CTkFrame(input_frame, fg_color="#16213E", corner_radius=10)
frame.pack(fill="x", pady=8, padx=15)
ctk.CTkLabel(frame, text="Job Title", font=("Arial", 16)).pack(side="left", padx=10, pady=10)
entries["Job Title"] = ctk.CTkEntry(frame, placeholder_text="Enter job title")
entries["Job Title"].pack(side="right", fill="x", expand=True, padx=10, pady=10)

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

# Right Side – Output Area
output_frame = ctk.CTkFrame(main_container, fg_color="#0F3460", corner_radius=15)
output_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

ctk.CTkLabel(output_frame, text="Prediction Results", font=("Arial Bold", 22)).pack(pady=15)

# White Prediction Display
output_textbox = ctk.CTkTextbox(
    output_frame,
    height=300,
    font=("Consolas", 14),
    fg_color="white",  # White background
    text_color="black"  # Black text
)
output_textbox.pack(fill="both", expand=True, padx=20, pady=20)
output_textbox.insert("end", "Predicted salary will appear here...\n")

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
S
