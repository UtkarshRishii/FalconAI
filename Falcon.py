# Falcon.py

from Backend.Body.Listen import recognize_speech
from Backend.Body.Speak import Speak
from Backend.Brain.FalBrain import create_assistant, chat_with_falcon

def main():
    assistant = create_assistant()

    while True:
        print("\n🎙️  Say something...")
        user_input = recognize_speech()

        if not user_input:
            continue

        # Get response from brain
        print("🧠 Thinking...")
        response = chat_with_falcon(user_input, assistant)
        print(f"🤖 Falcon: {response}")

        # Speak the response
        Speak(response)

if __name__ == "__main__":
    main()
