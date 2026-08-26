SYSTEM_PROMPT = f""" 
Task:
 - You will be provided with an input word: write the first 3 words you associate to it separated by a comma.
 - No additional output text is allowed. 

Constraints:
- no carriage return characters are allowed in the answers.
- answers should be as short as possible.
            
Example: 
Input: sea
Output: water,beach,sun
"""

def prepare_conversation(cue_list):
    conversation_list = []
    for cue in cue_list:
        conversation_list.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Input: {cue}\nOutput:"},
            ])
    return conversation_list
