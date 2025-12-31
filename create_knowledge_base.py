"""
Create Knowledge Base Documents for Ecobot RAG System
Run this script to generate plant care documents
"""

import os
from pathlib import Path

def create_knowledge_base():
    """Create comprehensive plant care documents"""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("Creating knowledge base documents...")

# Document 1: Snake Plant Guide
    snake_plant = """Snake Plant (Sansevieria) Care Guide


Common Causes of Damage:

- Overwatering (the most common cause)
- Bad soil drainage
- Fungal growth due to moisture
- Underwatering for long time
- Cold temperature exposure
- Physical breakage
- Pests

Solutions for Damaged Snake Plant:
- Let the soil dry completely before watering again
- Change to well-draining soil
- Reduce watering to every 2–3 weeks
- Keep plant in warm place
- Remove damaged parts using clean scissors
- Use neem oil weekly if pests or fungus appear
- Improve airflow around the plant

DAMAGED SNAKE PLANT SYMPTOMS:

1. Overwatering Damage:
- Leaves feel soft and mushy
- Yellow or brown leaves
- Bad smell from soil
- Roots look black and slimy

Solutions:
- Take the plant out of the pot
- Cut all rotten roots using clean scissors
- Let the plant dry for one full day
- Repot in fresh, fast-draining soil
- Water only once every 2–3 weeks

2. Fungal Infections:
- Brown or black spots on leaves
- White, powder-like coating
- Rust-colored spots

Solutions:
- Remove all infected leaves
- Spray with neem oil (light mixture)
- Keep air moving around the plant
- Avoid pouring water on leaves
- Use fungicide if problem keeps getting worse

3. Physical Damage:
- Torn or cracked leaves
- Brown edges
- Wrinkled, thin leaves (from underwatering)

Solutions:
- Trim damaged parts with clean scissors
- If wrinkled → water more regularly
- Keep a steady watering routine
- Keep humidity around 30–50%

3. Prevention Tips:
- Do not water unless the soil is fully dry
- Use a pot with drainage holes
- Keep humidity low to medium (30–50%)
- Avoid cold wind or sudden temperature changes
- Clean leaves regularly
- Avoid overcrowding plants
- Do not spray water on the leaves

4. What to Do If Snake Plant Is Dead:
- Remove the plant immediately
- Throw away the old soil (do not reuse)
- Clean pot using hot water or 10% bleach
- Check nearby plants for rot or fungus
- Replace with healthy plant
- Review greenhouse watering habits to avoid repeat problems
"""

# Document 2: Spider Plant Guide
    spider_plant = """Spider Plant (Chlorophytum) Care Guide

Common Causes of Damage:
- Low humidity causing leaf tip problems
- Water quality (tap water chemicals)
- Overwatering or poor drainage
- Too much direct sunlight
- Fungal leaf problems
- Pests (spider mites, aphids)
- Lack of nutrients

2. Solutions for Damaged Spider Plant:
- Use filtered or stored water
- Increase humidity to 40–60%
- Move plant to indirect light
- Water only when top soil feels dry
- Apply mild fungicide if needed
- Use neem oil for pests
- Trim damaged tips cleanly
- Fertilize monthly with gentle plant food

DAMAGED SPIDER PLANT SYMPTOMS:

1. Brown Leaf Tips:

Causes: Tap water chemicals, low humidity, too much fertilizer

Symptoms:
- Leaf tips turn brown and dry
- Brown color slowly spreads

Solutions:
- Use filtered or stored tap water
- Raise humidity to 40–60%
- Reduce fertilizer use
- Trim brown tips at an angle

2. Yellow or Pale Leaves:

Causes: Too much sun, not enough nutrients, overwatering

Symptoms:
- Leaves lose bright green color
- Leaves become limp or fully yellow

Solutions:
- Move plant to bright, indirect light
- Water only when top of soil feels dry
- Fertilize once a month during growing season
- Make sure pot drains well

3. Fungal and Bacterial Problems:
a) Leaf Spot Disease:
- Brown/black spots with yellow edges
- Spots grow and join together

Solutions:
- Remove the affected leaves
- Avoid wetting the leaves
- Use copper fungicide
- Increase airflow

4. Pest Damage:

Common pests: Spider mites, aphids, mealybugs, scale

Symptoms:
- Tiny insects on leaves
- Sticky residue
- Fine webbing
-Yellow or spotty leaves

Solutions:
- Rinse plant with water
- Apply neem oil weekly
- Use insecticidal soap
- Keep the plant isolated
- Use natural helpers like ladybugs

3. Prevention Tips:
- Keep soil slightly moist, never flooded
- Avoid placing plant under strong sunlight
- Repot every 2–3 years
- Clean off dust from leaves
- Avoid using strong chemical fertilizers
- Keep greenhouse air flowing

4. What to Do If Spider Plant Is Dead:
- Remove entire plant from greenhouse
- Check if small baby plants can be saved
- Throw away soil and sterilize pot
- Look for cause: water, humidity, pests, light
- Adjust greenhouse conditions
"""

 # Document 3: Aloe Vera Guide
    aloe_vera = """Aloe Vera Care Guide

Common Causes of Damage:
- Overwatering (most common for aloe vera)
- Too much direct sunlight (sunburn)
- Underwatering for long time
- Fungal or bacterial problems
- Cold temperatures
- Poor airflow

2. Solutions for Damaged Aloe Vera:
- Stop watering for 1–2 weeks
- Change soil to cactus/succulent type
- Move plant to bright but indirect light
- Trim damaged parts carefully
- Let plant dry before replanting
- Use sulfur-based fungicide for fungus
-Keep plant warm, above 13°C

DAMAGED ALOE VERA SYMPTOMS:

1. Overwatering Damage (most common):
Symptoms:
- Soft, mushy, see-through leaves
- Brown/black base
- Bad smell
- Rotten roots

Solutions:
- Stop watering right away
- Remove plant from pot
- Cut all rotten roots and leaves
- Let the plant dry 2–3 days
- Repot in dry cactus soil
- Water every 3–4 weeks only

2. Underwatering Damage:

Symptoms:
- Thin, wrinkled leaves
- Leaves curl inward
- Dry brown tips
- Leaves look deflated

Solutions:
- Water deeply once
- Follow a regular watering schedule
- Water only when soil is fully dry

3. Sunburn or Light Damage:

Symptoms:
- Leaves turn red, orange, or brown
- Dry spots
- Plant leans away from the light

Solutions:
- Move to bright, indirect sunlight
- Slowly introduce to stronger light
- Shade during hot hours
- Trim burnt parts if needed

4. Fungal & Bacterial Infections:

a) Aloe Rust (Fungal):
- Brown/black round spots
- Spots may spread

Solutions:
- Remove affected leaves
- Apply sulfur-based fungicide
- Lower humidity
- Increase airflow

b) Soft Rot (Bacterial):
- Sudden leaf collapse
- Wet, smelly patches
- Very fast damage

Solutions:
- Remove infected parts immediately
- Use bactericide if early
- Often impossible to save — prevention is best

3. Prevention Tips:
- Use soil that drains very fast
- Water every 3–4 weeks only
- Keep humidity low (30–40%)
- Do not spray water on leaves
- Keep plant away from cold windows
- Avoid placing plant in strong afternoon sun

4. What to Do If Aloe Vera Is Dead:
- Remove plant and check for healthy baby shoots
- Throw away soil immediately
- Clean pot with hot water
- Reduce watering schedule for all succulents in greenhouse
Keep new aloe in warm and dry area
"""

# Document 4: Common Greenhouse Problems
    common_problems = """General Greenhouse Care Guide

Common Causes of Plant Damage in Greenhouses:
- High humidity inside greenhouse
- Poor airflow
- Overwatering plants
- Not enough sunlight
- Too much direct sunlight
- Temperature too hot or too cold
- Pests spreading from plant to plant
- Fungal growth due to moisture

2. Solutions for Greenhouse Problems:
- Use fans to move air
- Open vents or windows regularly
- Keep plants spaced apart
- Use shade cloth if sunlight is too strong
- Use heater in winter
- Use humidifier or dehumidifier as needed
- Water plants in morning only
- Clean greenhouse floor weekly

3. Prevention Tips:
- Inspect all plants weekly
- Remove dead or weak plants immediately
- Sterilize tools before using
- Do not overcrowd plants
- Quarantine new plants for 1–2 weeks
- Use good quality soil
- Keep a temperature between 18°C and 26°C

4. What to Do When a Plant Dies:
- Remove plant completely
- Dispose soil safely
- Clean the pot
- Check nearby plants for similar issues
- Adjust watering or humidity if many plants are dying
- Replace with healthy plant only after cleaning are
"""

# Write all documents
    docs = {
        "snake_plant_guide.txt": snake_plant,
        "spider_plant_guide.txt": spider_plant,
        "aloe_vera_guide.txt": aloe_vera,
        "common_problems.txt": common_problems
    }


    for filename, content in docs.items():
        filepath = data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created: {filename}")
    
    print(f"\n✓ All {len(docs)} knowledge base documents created in 'data' folder!")
    
if __name__ == "__main__":
    create_knowledge_base()
    print("\n=== Knowledge Base Creation Complete ===")






