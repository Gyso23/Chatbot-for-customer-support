import json
import sys
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
import torch

# Load model data
def load_model_data(file_path):
    try:
        data = torch.load(file_path)
        return data["input_size"], data["hidden_size"], data["output_size"], \
               data['all_words'], data['tags'], data["model_state"]
    except Exception as e:
        print(f"Error loading model data: {e}")
        return None, None, None, None, None, None

# Function to get model output
def get_model_output(sentence, all_words, tags, model, device):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    return tag

if __name__ == "__main__":
    # Load model data
    input_size, hidden_size, output_size, all_words, tags, model_state = load_model_data("data.pth")
    
    if None in [input_size, hidden_size, output_size, all_words, tags, model_state]:
        exit()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Load intents data
    try:
        with open("intents.json", 'r') as file:
            intents_data = json.load(file)
    except Exception as e:
        print(f"Error loading intents data: {e}")
        exit()

    total_intents = len(intents_data['intents'])
    total_correct = 0
    total_wrong = 0

    # Perform Intent Recognition Test
    print("Intent Recognition Test Results:")
    for intent in intents_data['intents']:
        print(f"Intent: {intent['tag']}")
        for query in intent['queries']:
            predicted_intent = get_model_output(query, all_words, tags, model, device)
            print(f"Query: {query}")
            print(f"Predicted Intent: {predicted_intent}")
            user_input = input("Press Enter if correct, Space if wrong: ")
            if user_input == "":
                total_correct += 1
            elif user_input == " ":
                total_wrong += 1
            else:
                print("Invalid input. Skipping...")
            print()
    
    print("\nSummary:")
    print(f"Total Tags: {total_intents}")
    print(f"Total Correct: {total_correct}")
    print(f"Total Wrong: {total_wrong}")

