import json

PROMPT = f"""[Observation] <image>

[Instruction] {input['instruction']}

[History] {input['history']}

[Action options] {json.dumps(input['candidate'], indent=2)}
"""
