import random
import csv
import torch
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bot_name = "Lyla"
intents = None  # Define the intents variable
input_size, hidden_size, output_size, all_words, tags, model_state = None, None, None, None, None, None
# Function to load model data
def load_model_data(file_path):
    try:
        data = torch.load(file_path)
        return data["input_size"], data["hidden_size"], data["output_size"], \
               data['all_words'], data['tags'], data["model_state"]
    except Exception as e:
        print(f"Error loading model data: {e}")
        return None, None, None, None, None, None

# Function to get user input
def get_user_input():
    return input("You: ").lower()

# Function to handle exit commands
def handle_exit_commands(sentence, bot_name):
    exit_commands = ["quit", "exit", "bye", "thanks", "that's all thanks", "goodbye"]
    if sentence in exit_commands:
        print(f"{bot_name}: Goodbye! Have a great day!")
        return True
    return False

# Function to handle special cases
def handle_special_cases(sentence, bot_name):
    special_cases = {
        "I don't understand": "I apologize for any confusion. Please feel free to ask again.",
        "I have to go": "Alright! Take care and see you soon!"
    }
    if sentence in special_cases:
        print(f"{bot_name}: {special_cases[sentence]}")
        return True
    return False

# Function to get model output
def get_model_output(sentence, all_words, tags, model, device):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    return tag, prob.item()

# Function to generate response
def get_response(tag, prob, intents, bot_name, context=None):
    if prob > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if context:
                    for response in intent['responses']:
                        if "{context}" in response:
                            print(f"{bot_name}: {response.format(context=context)}")
                            return
                    else:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                else:
                    if tag == "order_status":  # Check if the tag is for tracking an order
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        order_number = input("Please enter your order number: ")
                        track_order(order_number)  # Call the function to track the order
                    elif tag == "update_address":  # Check if the tag is for updating the address
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        email = input("Please provide your email for verification: ")
                        order_number = input("Please enter your order number: ")
                        if verify_customer_email(order_number, email):  # Verify the provided email
                            current_address = fetch_order_info(order_number)['customer_address']
                            print(f"{bot_name}: Your current registered address is: {current_address}")
                            new_address = input("Please enter your new address: ")
                            print(f"{bot_name}: You entered the following address: {new_address}")
                            confirmation = input("Are you sure this is your new address? (yes/no): ")
                            if confirmation.lower() == "yes":
                                update_address(order_number, new_address)  # Call the function to update the address
                                print(f"{bot_name}: Your address has been updated.")
                            else:
                                print(f"{bot_name}: Address update cancelled.")
                        else:
                            print(f"{bot_name}: Email verification failed.")

                    else:
                        response = random.choice(intent['responses'])
                        if "{context}" in response:
                            context = intent["tag"]
                            print(f"{bot_name}: {response.format(context=context)}")
                        else:
                            print(f"{bot_name}: {response}")
    else:
        print(f"{bot_name}: I do not understand...")

# Function to fetch order information from CSV
def fetch_order_info(order_number):
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

# Function to handle tracking order logic
def track_order(order_number):
    order_info = fetch_order_info(order_number)
    if order_info:
        print("")
        print("Order Information:")
        print(f"Order Number: {order_info['order_number']}")
        print(f"Status: {order_info['status']}")
        print(f"Current Location: {order_info['current_location']}")
        print(f"Destination Address: {order_info['customer_address']}")
        print(f"Expected Delivery Date: {order_info['expected_delivery_date']}")
        print("")
    else:
        print("Order not found.")

# Function to verify customer_email from CSV
def verify_customer_email(order_number, customer_email):
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


#Function to fetch customer's address
def fetch_customer_address(email):
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


# Function to update address in CSV
def update_address(order_number, new_address):
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
        
# Function to load model data
def load_model_data(file_path):
    global input_size, hidden_size, output_size, all_words, tags, model_state
    try:
        data = torch.load(file_path)
        input_size, hidden_size, output_size, all_words, tags, model_state = (
            data["input_size"], 
            data["hidden_size"], 
            data["output_size"],
            data['all_words'], 
            data['tags'], 
            data["model_state"]
        )
    except Exception as e:
        print(f"Error loading model data: {e}")

if __name__ == "__main__":
    input_size, hidden_size, output_size, all_words, tags, model_state = load_model_data("data.pth")
    
    if None in [input_size, hidden_size, output_size, all_words, tags, model_state]:
        exit()

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    load_model_data("data.pth")

    # Load intents data
    try:
        with open("intents.json", 'r') as file:
            intents = json.load(file)
    except Exception as e:
        print(f"Error loading intents data: {e}")
        exit()

    bot_name = "Lyla"
    context = None  # Initialize context variable

    print("Hello! I am Lyla from CoffeeMugs. What do you need? (type 'quit', 'exit', or 'bye' to exit)")

    while True:
        sentence = get_user_input()

        if handle_exit_commands(sentence, bot_name) or handle_special_cases(sentence, bot_name):
            break

        tag, prob = get_model_output(sentence, all_words, tags, model, device)
        get_response(tag, prob, intents, bot_name, context)

