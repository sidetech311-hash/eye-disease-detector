import os
required_folders = ['ageDegeneration', 'cataract', 'diabetes', 'glaucoma', 'hypertension', 'myopia', 'normal', 'others']
for folder in required_folders:
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"{folder}: {len(files)}")
    else:
        print(f"{folder}: MISSING")
