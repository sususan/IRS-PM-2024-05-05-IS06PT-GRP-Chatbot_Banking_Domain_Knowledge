import pandas as pd
qa_df = pd.read_excel("./intents_utterances_answers.xlsx")
answers = { intent : {'utterance': question, 'answer': answer}  
           for intent, question, answer in zip(qa_df['intent'], qa_df['utterance'], qa_df['answers'])}

print("loading ...")
import intentmodel
from intentmodel import nlu
from intentmodel import clear_screen

def main():
    clear_screen()

    while True:
        print()
        user_input = input("Welcome to DBS!  How can I assist you today? Ask me anything about cashback, account qualifications, or any other banking queries you have. Let's get started! Enter your text (type 'exit' to quit): \n")

        if user_input.lower() == 'exit':
            print("Thank you for trying out our DBS Chatbot for NUS practice module")
            break

        intent = nlu.get_intent(user_input.lower()  )
        utterance = answers[intent]['utterance']
        answer = answers[intent]['answer']

        # Display the processed message
        print(f"Do you mean to ask:\n {utterance}?" 
              f"\n\n"
              f"{answer}"
            )


if __name__ == '__main__':
    main()
