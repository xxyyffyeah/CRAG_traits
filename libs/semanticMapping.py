import json
import os
from dotenv import load_dotenv
from model import get_response

# Load environment variables from .env file
load_dotenv()

def add_semantic_mapping_to_traits():
    """
    Add 'Avoid' and 'Prefer' attributes to each trait in traits.json using GPT-5.1
    """
    # Read the traits.json file
    traits_path = '/home/xuyifan/rank_grpo/baselines/CRAG/traits.json'
    with open(traits_path, 'r') as f:
        traits_data = json.load(f)

    # Prepare the prompt template for GPT-5.1
    prompt_template = """Given the following user trait:
Trait: {trait}
Description: {description}

Based on this trait, suggest 2-4 content themes to AVOID and 2-4 content themes to PREFER for this user when recommending movies or media content.

For example, for a domestic violence survivor:
- Avoid: Domestic Abuse, Child Trauma, Gaslighting
- Prefer: Escapism, Empowerment

Provide your response in the following JSON format only, with no additional text:
{{
    "avoid": ["theme1", "theme2", "theme3"],
    "prefer": ["theme1", "theme2", "theme3"]
}}"""

    results = []
    existing_cache = {}

    # Process each category and trait
    index = 0
    total_traits = sum(len(traits) for traits in traits_data.values())

    print(f"Processing {total_traits} traits across {len(traits_data)} categories...")

    for category, traits in traits_data.items():
        print(f"\nProcessing category: {category}")

        for trait_obj in traits:
            trait = trait_obj['trait']
            description = trait_obj['description']

            print(f"  [{index + 1}/{total_traits}] Processing: {trait}")

            # Prepare the text for the API call
            text = {
                'trait': trait,
                'description': description
            }

            # Call GPT-5.1 API
            get_response(
                index=index,
                text=text,
                prompt=prompt_template,
                model='gpt-5.1',  # Using GPT-5.1
                temperature=0.7,
                max_tokens=300,
                results=results,
                EXSTING=existing_cache
            )

            index += 1

    # Sort results by index to maintain order
    results.sort(key=lambda x: x['index'])

    # Process results and update traits_data
    index = 0
    for category, traits in traits_data.items():
        for trait_obj in traits:
            if index < len(results):
                result = results[index]

                # Extract the response content
                try:
                    response_text = result['resp']['choices'][0]['message']['content']

                    # Parse the JSON response
                    # Remove markdown code blocks if present
                    response_text = response_text.strip()
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.startswith('```'):
                        response_text = response_text[3:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()

                    semantic_mapping = json.loads(response_text)

                    # Add the avoid and prefer fields to the trait object
                    trait_obj['avoid'] = semantic_mapping.get('avoid', [])
                    trait_obj['prefer'] = semantic_mapping.get('prefer', [])

                    print(f"  ✓ Added mapping for {trait_obj['trait']}")

                except Exception as e:
                    print(f"  ✗ Error parsing response for {trait_obj['trait']}: {e}")
                    print(f"    Response: {result['resp']}")
                    # Add empty lists as fallback
                    trait_obj['avoid'] = []
                    trait_obj['prefer'] = []

            index += 1

    # Write the updated traits back to the file
    output_path = '/home/xuyifan/rank_grpo/baselines/CRAG/traits_with_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(traits_data, f, indent=2)

    print(f"\n✓ Successfully created {output_path} with semantic mappings!")
    print(f"  Processed {total_traits} traits")

    return traits_data

if __name__ == '__main__':
    # Set up OpenAI API key from environment
    import openai

    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        openai.api_key = api_key
        print("✓ Loaded OpenAI API key from .env file")
        add_semantic_mapping_to_traits()
    else:
        print("✗ Error: OPENAI_API_KEY not found in .env file")
        print("Please ensure .env file contains: OPENAI_API_KEY='your-api-key'")
