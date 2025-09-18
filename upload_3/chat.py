import base64
from openai import OpenAI

def encode_image(image_path):
    """Read an image and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    client = OpenAI()

    print("Chat with GPT-4o (type 'exit' to quit)\n")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Format: send_image image.png Your question here
        if user_input.startswith("send_image "):
            parts = user_input.split(" ", 2)
            if len(parts) < 2:
                print("⚠️ Usage: send_image <image_path> [your message]")
                continue

            image_path = parts[1]
            text_prompt = parts[2] if len(parts) > 2 else "Analyze this image."

            base64_image = encode_image(image_path)

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",   # or "gpt-4o"
            messages=messages
        )

        reply = response.choices[0].message.content
        print(f"GPT: {reply}\n")

        messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
