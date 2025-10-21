import customtkinter as ctk
import chatbot
import linear_regression as lr

def button_callback():
    text = chat_entry.get()
    chat_display.insert("end","You: "+text+"\n")
    chat_display.insert("end","Chatbot: "+chatbot.chat(text))
    chat_entry.delete(0, "end")
    

def button_callback2():
    age = int(entries["Age"].get())
    gender = entries["Gender"].get()
    education_level = entries["Education Level"].get()
    experience = int(entries["Experience"].get())
    job_title = entries["Job Title"].get()
    salary = 0

    user_attributes = [age, gender,education_level,job_title,experience,salary]
    prediction = lr.predict(user_attributes)
    prediction = round(prediction[0],2)
    
    text="This person should be earning: " + str(prediction)
    lblDisplayLeft.configure(text=text)

    
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Create main window
app = ctk.CTk()
app.title("AI Salary Predictor")

app.state("zoomed")  # Open fullscreen
app.geometry("1300x1500")
app.configure(fg_color="#1A1A2E")

#  HEADER
header_frame = ctk.CTkFrame(app, fg_color="#1A1A2E", corner_radius=0)
header_frame.pack(fill="x")

header_label = ctk.CTkLabel(
    header_frame,
    text="Earnlytics: AI Salary Predictor",
    font=("Arial Rounded MT Bold", 28),
    text_color="white"
)
header_label.pack(padx=10, pady=10)

#CONTAINERS
#Normal Graphic User Interface (left)
employee_frame = ctk.CTkFrame(app, fg_color="#16213E", corner_radius=20)
employee_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)
employee_frame.pack_propagate(False)
ctk.CTkLabel(employee_frame, text="Employee", font=("Arial Bold", 22)).pack(pady=15)

#Chatbot user interface
chatbot_frame = ctk.CTkFrame(app, fg_color="#16213E", corner_radius=20)
chatbot_frame.pack(side = "right",fill="both", expand=True, padx=20, pady=20)
chatbot_frame.pack_propagate(False)
ctk.CTkLabel(chatbot_frame, text="Chatbot", font=("Arial Bold", 22)).pack(pady=15)

##LEFT GRAPHIC USER
entries = {}
# --- Age ---
age_frame = ctk.CTkFrame(employee_frame, fg_color="#16213E", corner_radius=10)
age_frame.pack(fill="x", pady=0, padx=15)
ctk.CTkLabel(age_frame , text="Age", font=("Arial", 16)).pack(side="left", padx=10, pady=10)
entries["Age"] = ctk.CTkEntry(age_frame , placeholder_text="Enter age")
entries["Age"].pack(side="right", fill="x", expand=True, padx=10, pady=5)

# --- Gender Dropdown ---
frame = ctk.CTkFrame(employee_frame, fg_color="#16213E", corner_radius=10)
frame.pack(fill="x", pady=0, padx=15)
ctk.CTkLabel(frame, text="Gender", font=("Arial", 16)).pack(side="left", padx=10, pady=10)
entries["Gender"] = ctk.CTkOptionMenu(frame, values=["Male", "Female", "Other"])
entries["Gender"].pack(side="right", fill="x", expand=True, padx=10, pady=5)

# --- Education Dropdown ---
frame = ctk.CTkFrame(employee_frame, fg_color="#16213E", corner_radius=10)
frame.pack(fill="x", pady=0, padx=15)
ctk.CTkLabel(frame, text="Education Level", font=("Arial", 16)).pack(side="left", padx=10, pady=10)
entries["Education Level"] = ctk.CTkOptionMenu(frame, values=["High School", "Diploma", "Bachelor's", "Master's", "PhD"])
entries["Education Level"].pack(side="right", fill="x", expand=True, padx=10, pady=10)

# --- Experience ---
frame = ctk.CTkFrame(employee_frame, fg_color="#16213E", corner_radius=10)
frame.pack(fill="x", pady=0, padx=15)
ctk.CTkLabel(frame, text="Years of Experience", font=("Arial", 16)).pack(side="left", padx=10, pady=10)
entries["Experience"] = ctk.CTkEntry(frame, placeholder_text="Enter years of experience")
entries["Experience"].pack(side="right", fill="x", expand=True, padx=10, pady=10)

# --- Job Title ---
frame = ctk.CTkFrame(employee_frame, fg_color="#16213E", corner_radius=10)
frame.pack(fill="x", pady=0, padx=15)
ctk.CTkLabel(frame, text="Job Title", font=("Arial", 16)).pack(side="left", padx=10, pady=10)
entries["Job Title"] = ctk.CTkEntry(frame, placeholder_text="Enter job title")
entries["Job Title"].pack(side="right", fill="x", expand=True, padx=10, pady=10)

predict_button = ctk.CTkButton(
    employee_frame,text="Predict Salary",
    height=40,width=200,
    font=("Arial Rounded MT Bold", 16),
    fg_color="#533483",
    hover_color="#6D44A0",
    command=button_callback2
    
)

predict_button.pack(pady=25)

lblDisplayLeft = ctk.CTkLabel(
    employee_frame,
    height=300,
    font=("Consolas", 14),
    fg_color="#42414D",  # White background
    text_color="White", # Black text
    text = "",
    anchor="nw",
    corner_radius=10
    
)
lblDisplayLeft.pack(fill="both", expand=True, padx=20, pady=20)
lblDisplayLeft.configure(text = "Predicted salary will appear here...\n")


# = = = = =END OF LEFT GRAPHIC USER INTERFACE AND START OF RIGHT CHATBOT INTERFACE = = =

ctk.CTkLabel(
    chatbot_frame,
    text="🤖 Chat with Earnlytics Assistant",
    font=("Arial Bold", 18),
    text_color="white"
).pack(anchor="w", padx=20, pady=(10, 0))

chat_display = ctk.CTkTextbox(chatbot_frame, height=400, font=("Consolas", 13))
chat_display.pack(fill="x", padx=20, pady=10)

# Auto greeting
chat_display.insert("end", "Earnlytics :Hello! I’m here to help you navigate the AI Salary Predictor.\n")
chat_display.insert("end", "Feel free to ask me questions about the system or salary insights.\n\n")
chat_display.tag_config("bot_color", foreground="#00FFFF")  # light cyan



chat_entry = ctk.CTkEntry(chatbot_frame, placeholder_text="Type your message here...")

chat_entry.pack(side="left", fill="x", expand=True, padx=(20, 10), pady=(0, 20))

send_button = ctk.CTkButton(
    chatbot_frame,
    text="➤",
    width=100,
    height=35,
    fg_color="#533483",
    hover_color="#6D44A0",
    font=("Arial Rounded MT Bold", 14),
    command=button_callback
)

send_button.pack(side="right", padx=(0, 20), pady=(0, 20))

app.mainloop()
