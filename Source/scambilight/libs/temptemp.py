import json

plop = {"id": "something","width": 640,
            "height": 480,
            "fish_eye_circle": 600}

print(json.dumps(plop))

farts = [[18, 114], [587, 173], [431, 301], [176, 285]]

print(json.dumps(farts))