import tkinter as tk
from tkinter import scrolledtext, simpledialog
import random
import csv
import torch
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ChatGUI:
    def __init__(self, master):
        self.master = master
        master.title("Chat with Lyla")
        master.geometry("400x500")

        self.chat_history = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=40, height=20)
        self.chat_history.pack(padx=10, pady=10)

        self.input_field = tk.Entry(master, width=30)
        self.input_field.pack(padx=10, pady=10)
        self.input_field.bind("<Return>", self.send_message)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(padx=10, pady=10)

        self.bot_name = "Lyla"
        self.intents = self.load_intents("intents.json")  # Load intents data
        self.input_size, self.hidden_size, self.output_size, self.all_words, self.tags, self.model_state = self.load_model_data("data.pth")
        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(device)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

    def load_intents(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading intents data: {e}")
            return None

    def load_model_data(self, file_path):
        try:
            data = torch.load(file_path)
            return (
                data["input_size"],
                data["hidden_size"],
                data["output_size"],
                data['all_words'],
                data['tags'],
                data["model_state"]
            )
        except Exception as e:
            print(f"Error loading model data: {e}")
            return None, None, None, None, None, None

    def send_message(self, event=None):
        user_input = self.input_field.get()
        self.input_field.delete(0, tk.END)
        self.display_message("You", user_input)

        # Call the chatbot logic to generate a response
        self.generate_bot_response(user_input)

    def generate_bot_response(self, user_input):
        tag, prob = self.get_model_output(user_input)
        self.get_response(tag, prob)

    def get_model_output(self, sentence):
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        return tag, prob.item()

    def get_response(self, tag, prob):
        if prob > 0.75:
            for intent in self.intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    self.display_message(self.bot_name, response)

                    if tag == "order_status":
                        self.track_order_gui()
                    elif tag == "update_address":
                        self.update_address_gui()

                    return
        else:
            self.display_message(self.bot_name, "I do not understand, please rephrase your question so I can better assist you.")

    def display_message(self, sender, message):
        if sender == "You":
            self.chat_history.insert(tk.END, f"{message}\n", 'user_message')
        else:
            self.chat_history.insert(tk.END, f"{message}\n", 'bot_message')
        self.chat_history.see(tk.END)

        # Apply styles to the text based on sender
        self.chat_history.tag_config('user_message', justify='right', foreground='white', background='blue')
        self.chat_history.tag_config('bot_message', justify='left', foreground='black', background='lightgrey')


    def track_order_gui(self):
        order_number = simpledialog.askstring("Track Order", "Please enter your order number:")
        if order_number:
            order_info = self.fetch_order_info(order_number)
            if order_info:
                message = (
                    f"Order Number: {order_info['order_number']}\n"
                    f"Status: {order_info['status']}\n"
                    f"Current Location: {order_info['current_location']}\n"
                    f"Destination Address: {order_info['customer_address']}\n"
                    f"Expected Delivery Date: {order_info['expected_delivery_date']}\n"
                )
                self.display_message(self.bot_name, message)
            else:
                self.display_message(self.bot_name, "Order not found.")
        else:
            self.display_message(self.bot_name, "Order number cannot be empty.")

    def update_address_gui(self):
        order_number = simpledialog.askstring("Update Address", "Please enter your order number:")
        email = simpledialog.askstring("Update Address", "Please provide your email for verification:")
        if order_number and email:
            if self.verify_customer_email(order_number, email):
                current_address = self.fetch_customer_address(email)
                new_address = simpledialog.askstring("Update Address", f"Your current address is {current_address}. Please enter your new address:")
                if new_address:
                    confirmation = simpledialog.askstring("Update Address", f"You entered the following address: {new_address}. Are you sure this is your new address? (yes/no):")
                    if confirmation.lower() == "yes":
                        self.update_address(order_number, new_address)
                        self.display_message(self.bot_name, "Your address has been updated.")
                    else:
                        self.display_message(self.bot_name, "Address update cancelled.")
            else:
                self.display_message(self.bot_name, "Email verification failed.")
        else:
            self.display_message(self.bot_name, "Order number and email cannot be empty.")

    def fetch_order_info(self, order_number):
        try:
            with open('orders.csv', mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['order_number'] == order_number:
                        return row
            return None  # Order number not found
        except Exception as e:
            print(f"Error fetching order information: {e}")
            return None

    def verify_customer_email(self, order_number, customer_email):
        try:
            with open('orders.csv', mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['order_number'] == order_number and row['customer_email'] == customer_email:
                        return True
            return False
        except Exception as e:
            print(f"Error verifying email: {e}")
            return False

    def fetch_customer_address(self, email):
        try:
            with open('orders.csv', mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['customer_email'] == email:
                        return row['customer_address']
            return None  # Customer email not found
        except Exception as e:
            print(f"Error fetching customer address: {e}")
            return None

    def update_address(self, order_number, new_address):
        try:
            rows = []
            with open('orders.csv', mode='r') as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames
                for row in reader:
                    if row['order_number'] == order_number:
                        row['customer_address'] = new_address
                    rows.append(row)
            fieldnames.append('customer_address')  # Add 'address' field to fieldnames list
            with open('orders.csv', mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            print(f"Error updating address: {e}")


root = tk.Tk()
chat_gui = ChatGUI(root)
root.mainloop()