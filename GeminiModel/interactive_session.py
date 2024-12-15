from GeminiModel.text_to_speech import speak_text
import google.generativeai as genai

genai.configure(api_key="AIzaSyCoFnXOxt8Hau3kQZuVTX3llO7R_bQh5ss")
model1 = genai.GenerativeModel("gemini-1.5-flash")

def interactive_questionnaire(questions):
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        speak_text(question)

        # Get user's response
        user_response = input("\nYour Answer: ").strip()

        # Generate a follow-up suggestion or assistive reply
        suggestion_prompt = (
            f"You are responding to the question: '{question}'. "
            f"The user's answer was: '{user_response}'. "
            "Provide a short, helpful suggestion or response to assist the user. "
            "Then, mention that you will move to the next question."
        )
        follow_up = model1.generate_content(suggestion_prompt).text.strip()

        # Display and speak the response
        print(f"\nAgent: {follow_up}")
        speak_text(follow_up)

    print("\nAll questions have been answered. Thank you!")
    speak_text("All questions have been answered. Thank you!")
