from dotenv import load_dotenv

import os
import anthropic
import json
import time

with open("data/seed_data/infp_features.json", "r") as f:
    infp_data = json.load(f)
    infp_data = list(infp_data.values())

# TODO: Actually use the situation data
with open("data/seed_data/situation.json", "r") as f:
    situation = json.load(f)
    situation = list(infp_data.values())

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)

infp_dialogue_dataset = list()
for i in range(5):
    feature = infp_data[i]

    # TODO: Actually use the situation data
    situation = situation[i]

    system_prompt='''
    Create in the following format where the conversation utterances are surrounded by quotes:
    "User": "Detailed utterance 1"
    "INFP": "Detailed utterance 2"

    In this conversation, "User" is an arbitrary person and "INFP" is the INFP character. They have already been introduced and are familiar with each other. The conversation should be a back-and-forth dialogue. Each person should speak between 4 to 10 times.
    Following this format is mandatory and very important.
    '''

    prompt = f'''Begin by exploring the broad characteristics of the INFP personality type, known for their deep empathy, creativity, introspection, and preference for meaningful connections over superficial interactions. INFPs are often seen as idealistic and passionate, with a unique ability to see the good in people and situations. They are driven by their values and seek to understand themselves and others on a profound level. INFPs are also known for their adaptability and open-mindedness, allowing them to navigate complex emotional landscapes with grace.

    Given these traits, generate a dialogue between "User" and "INFP" that showcases the specific INFP characteristic:
    “{feature}"
    Focusing on this characteristic is very very very important. The dialouge should well reflect this trait.
    This conversation is happening in this situation:
    "{situation}"
    '''
    success = False

    while success == False:
        try:
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                system=system_prompt,
                max_tokens=1000,
                temperature=1,
                messages=[
                    {"role": "user", "content": f"{prompt}"}
                ]
            )
            success = True
        except Exception as e:
            print(e)
            time.sleep(20)
    dialogue = message.content[0].text.split("\n\n")
    infp_dialogue_dataset.append({"feature": feature, "dialogue": dialogue})
    print(f'finished_{i}')


with open("data/generated_data/infp_feature_dialogues.json", "w") as f:
    json.dump(infp_dialogue_dataset, f, indent=4)