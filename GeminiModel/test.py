import concurrent.futures
from GeminiModel.image_caption import generate_image_caption, generate_ene_caption
from GeminiModel.questionnaire import generate_questionnaire_with_gemini
from GeminiModel.interactive_session import interactive_questionnaire

def generate_captions_in_parallel(image_paths):
    # Define prompts for each image
    prompts = [
        "<grounding>Describe the condition and objects in this scene in detail.",
        "<grounding>Describe the overall environment, including objects, conditions, and any notable features.",
        #"<grounding>Describe the overall environment, including objects, conditions, and any notable features. after changing the environment state"
    ]

    tasks = [
        (generate_image_caption, image_paths[0], prompts[0]),
        (generate_ene_caption,image_paths[1], prompts[1]),
        # (generate_ene2_caption,image_paths[2], prompts[2])
    ]

    
    # Use ThreadPoolExecutor to run caption generation in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(func, image, prompt) for func, image, prompt in tasks]
        captions = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return captions


def main(emot,context,eeg):
    eeg_data = {"emotion":eeg}
    #gaze_object = image
    emotion=emot
    voice_text = context
    image_paths = ["GeminiModel\s1.webp", "cropped_image.jpg"]

    print("Generating captions for the images in parallel...")
    captions = generate_captions_in_parallel(image_paths)

    image_caption = captions[0]
    env_caption = captions[1]
    #env2_caption = captions[2]

    print("Generating questionnaire...")
    questionnaire = generate_questionnaire_with_gemini(eeg_data, emotion, voice_text, image_caption, env_caption)
    questions = questionnaire.strip().split("\n")
    
    print("\nGenerated Questions:\n", "\n".join(questions))

    print("\nStarting the interactive session...\n")
    interactive_questionnaire(questions)

if __name__ == "__main__":
    main(context)
