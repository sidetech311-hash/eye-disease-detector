from PIL import Image, ImageDraw, ImageFont
import os

def generate_professional_logo():
    # 1. Create a 500x500 canvas with a transparent background
    size = (500, 500)
    img = Image.new('RGBA', size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # 2. Define Colors
    med_blue = (26, 115, 232)      # #1a73e8
    retina_orange = (232, 115, 26) # #e8731a
    dark_gray = (51, 51, 51)

    # 3. Draw Outer Glowing Circle
    draw.ellipse([50, 50, 450, 450], outline=med_blue, width=15)

    # 4. Draw Stylized Eye Iris
    # Outer Iris
    draw.ellipse([120, 120, 380, 380], fill=med_blue)
    # Inner Pupil
    draw.ellipse([200, 200, 300, 300], fill=(0, 0, 0))

    # 5. Draw "Neural/Retinal" Vessels (AI Patterns)
    # Drawing 4 branching lines to represent the AI/Retina connection
    draw.line([250, 250, 400, 150], fill=retina_orange, width=5)
    draw.line([250, 250, 100, 150], fill=retina_orange, width=5)
    draw.line([250, 250, 400, 350], fill=retina_orange, width=5)
    draw.line([250, 250, 100, 350], fill=retina_orange, width=5)

    # Dots at the end of lines to look like "Nodes"
    for pos in [(400, 150), (100, 150), (400, 350), (100, 350)]:
        draw.ellipse([pos[0]-8, pos[1]-8, pos[0]+8, pos[1]+8], fill=retina_orange)

    # 6. Draw "Scanning" Line
    draw.rectangle([50, 245, 450, 255], fill=(232, 115, 26, 150))

    # 7. Save the file
    img.save("eyecare_ai_logo.png")
    print("✅ Logo generated successfully as 'eyecare_ai_logo.png'")

if __name__ == "__main__":
    generate_professional_logo()
