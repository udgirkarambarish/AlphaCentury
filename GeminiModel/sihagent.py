import google.generativeai as genai
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import pyttsx3  # Text-to-Speech library

# Step 1: Configure Gemini API
genai.configure(api_key="AIzaSyCoFnXOxt8Hau3kQZuVTX3llO7R_bQh5ss")
model1 = genai.GenerativeModel("gemini-1.5-flash")

# Step 2: Text-to-Speech Setup
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()

# Step 3: Image Caption Generator
def generate_image_caption(image_path, prompt="<grounding>Describe the condition and objects in this scene in detail."):
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Generate Caption
    generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
            temperature=0.9 
    )

    # Decode Output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()

def generate_ene_caption(image_path, prompt="<grounding>Describe the overall environment, including objects, conditions, and any notable features."):
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Generate Caption
    generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
            temperature=0.9 
    )

    # Decode Output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()

# Step 4: Generate Questionnaire using Gemini AI
def generate_questionnaire_with_gemini(eeg_data, gaze_object, voice_text, image_caption, env_caption):
    # Combine the static inputs into a prompt
    input_prompt = (
        f"Your emotional state is '{eeg_data['emotion']}' and your focus level is '{eeg_data['focus']}'. "
        f"You are focusing on '{gaze_object}' which is described as: {image_caption}. "
        f"The environment around you is described as: {env_caption}. "
        f"You also mentioned: '{voice_text}'. "
        "Based on this information, create 4 short and precise questions to assist or engage you. "
        "Questions should be direct, addressing you personally. Do not include any additional explanations or brackets. "
        "Format the questions as a numbered list."
    )
    
    # Generate output from Gemini AI
    response = model1.generate_content(input_prompt).text.strip()
    return response


def interactive_questionnaire(questions):
    """
    Conducts an interactive questionnaire session with the user.
    Asks questions one by one and provides suggestions or follow-ups.
    """
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        speak_text(question)  # Speak the question aloud

        # Get user's response
        user_response = input("\nYour Answer: ").strip()

        # Generate a follow-up suggestion or assistive reply based on the user's response
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


# Step 5: Main Function to Integrate Components
def main():
    # Static Input Data
    eeg_data = {"emotion": "focused", "focus": "high"}
    gaze_object = "a book"
    voice_text = "tasks for today"
    image_path = "env.webp"  # Focal image path
    image_pathevv = "focus.webp"  # Environment image path

    # Generate Captions
    print("Generating caption for the focal image...")
    image_caption = generate_image_caption(image_path)

    print("Generating caption for the environment image...")
    env_caption = generate_ene_caption(image_pathevv)

    # Generate Questionnaire
    print("Generating questionnaire...")
    questionnaire = generate_questionnaire_with_gemini(eeg_data, gaze_object, voice_text, image_caption, env_caption)
    questions = questionnaire.strip().split("\n")
    
    print("\nGenerated Questions:\n", "\n".join(questions))

    # Conduct the interactive session
    print("\nStarting the interactive session...\n")
    interactive_questionnaire(questions)

# Run Main
if __name__ == "__main__":
    main()
