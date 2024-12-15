
import google.generativeai as genai
genai.configure(api_key="AIzaSyCoFnXOxt8Hau3kQZuVTX3llO7R_bQh5ss")
model1 = genai.GenerativeModel("gemini-1.5-flash")
def generate_questionnaire_with_gemini(eeg_data,voice_text,voice_emotion, image_caption, env_caption):
    # Combine the static inputs into a prompt
    input_prompt = (
    f"Person's emotional state is '{eeg_data['emotion']}"
    f"The environment around you is described as: {env_caption}. "
    f"Person is looking at something described as: {image_caption}. "
    #f"This is the environment after the event: {image_caption} changed information as follows: {env2_caption}. "
    f"The emotions in voice is described as: {voice_emotion} and context of the  voice is described as: {voice_text}. "
    "Based on this information, create 4 short and precise questions to assist or engage the person. "
    "Questions should be direct, addressing you personally. Do not include any additional explanations or brackets. "
    "Format the questions as a numbered list. "
    "Prioritize the questions based on all user parameters mentioned above, and give the probability for each question being asked first based on their condition."
   )
    # Generate output from Gemini AI
    response = model1.generate_content(input_prompt).text.strip()
    return response

