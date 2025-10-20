import tkinter as tk
from tkinter import scrolledtext
from tkinter import END



# --- Create main window ---
root = tk.Tk()
root.title("Chatbot 💬")
root.geometry("500x600")
root.resizable(False, False)

# --- Chat display area ---
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=25, font=("Arial", 12))
chat_window.pack(padx=10, pady=10)
chat_window.config(state=tk.DISABLED)

# --- Message entry box ---
frame = tk.Frame(root)
frame.pack(pady=5)

user_input = tk.Entry(frame, width=40, font=("Arial", 12))
user_input.pack(side=tk.LEFT, padx=10)
user_input.bind("<Return>", send_message)

send_button = tk.Button(frame, text="Send", font=("Arial", 12), bg="#4CAF50", fg="white", command=send_message)
send_button.pack(side=tk.LEFT)

# --- Styling tags for messages ---
chat_window.tag_config("user", foreground="blue")
chat_window.tag_config("bot", foreground="green")

# --- Run the app ---
root.mainloop()
