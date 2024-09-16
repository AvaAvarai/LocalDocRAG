import random

class AssistantChat:
    def __init__(self):
        self.greetings = [
            "Hello! How can I assist you with your documents today?",
            "Welcome! What information would you like to retrieve from your documents?",
            "Greetings! How may I help you analyze your documents?",
            "Hi there! What would you like to know about your documents?"
        ]
        self.responses = {
            "default": [
                "I see. What specific information are you looking for in your documents?",
                "Understood. Can you provide more details about what you're searching for?",
                "Certainly. Which part of your documents would you like me to focus on?",
                "Thank you for your query. Is there a particular topic you'd like to explore further?",
                "I appreciate your question. How else can I assist you with your document analysis?"
            ],
            "goodbye": [
                "It was a pleasure assisting you with your documents. Have a great day!",
                "Thank you for using our document analysis service. Feel free to return if you need more help!",
                "Goodbye! I hope I was able to help you find the information you needed. Take care!"
            ]
        }

    def greet(self):
        return random.choice(self.greetings)

    def respond(self, user_input):
        user_input = user_input.lower()
        
        if "bye" in user_input or "goodbye" in user_input:
            return random.choice(self.responses["goodbye"])
        
        return random.choice(self.responses["default"])

def main():
    assistant = AssistantChat()
    print("Assistant:", assistant.greet())

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "goodbye", "exit", "quit"]:
            print("Assistant:", assistant.respond(user_input))
            break
        print("Assistant:", assistant.respond(user_input))

if __name__ == "__main__":
    main()
