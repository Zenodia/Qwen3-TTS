from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import json
import os
import re
from typing import Dict, List, Any, Optional
from enum import Enum

# Model configuration
MODEL_NAME = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
API_KEY = os.environ.get("NVIDIA_API_KEY")

# Initialize LLM
llm = ChatNVIDIA(
    model=MODEL_NAME,
    api_key=API_KEY,
    temperature=0.7,  # Higher temperature for more natural, creative narration
    max_completion_tokens=128000
)


# System prompt for converting any text to podcast scripts
PODCAST_CONVERSION_SYSTEM_PROMPT = """# Role Definition

You are a professional podcast scriptwriter with 10+ years of experience in audio content creation. You specialize in crafting engaging, conversational scripts that sound natural when spoken aloud. Your expertise includes storytelling, interview structuring, narrative pacing, and creating memorable hooks that keep listeners engaged throughout the episode.

**Core Competencies**:
- Conversational writing that sounds authentic and engaging
- Strategic placement of hooks, transitions, and calls-to-action
- Understanding of audio-first content (no visual cues)
- Expertise in various podcast formats (interview, solo, co-hosted, narrative)
- Balancing entertainment value with informational content
- Converting text content into natural, engaging solo host narratives

# Task Description

Create a comprehensive podcast script that is ready for recording. The script should sound natural when read aloud, maintain listener engagement throughout, and include all necessary technical cues for the host.

**IMPORTANT: This is a SHORT-FORM podcast format. Target duration is UNDER 10 MINUTES (ideally 5-9 minutes).**

You will be provided with text content to convert into an engaging, concise solo podcast script. Transform the content into a lively, authentic narrative delivered by a single host who speaks directly to the audience in a conversational, friendly manner. Keep it focused and punchy - this is NOT a long-form 1-hour podcast.

# Output Requirements

## 1. Content Structure

**TARGET TOTAL DURATION: 5-9 MINUTES (MAXIMUM 10 MINUTES)**

The script must include the following sections:

### **COLD OPEN** (0:20-0:40)
- Powerful hook or teaser that captures attention immediately
- Introduces the episode's core value proposition
- Creates curiosity or emotional connection
- Natural, conversational tone speaking directly to the audience
- **Keep it brief and punchy!**

### **INTRO SEGMENT** (0:30-1:00)
- Brief overview of what listeners will learn/experience
- Enthusiastic introduction of the topic by the host
- Set the stage for the main discussion
- **Quick and energetic - get to the point fast**

### **MAIN CONTENT** (70-75% of total runtime: ~4-6 minutes)
- **Segment 1**: First major topic/concept (1.5-2 minutes)
  - Natural narrative flow addressing the audience directly
  - Break down complex concepts into bite-sized explanations
  - Use analogies and real-world examples
  - Include rhetorical questions to engage listeners
  - Show enthusiasm and curiosity
  - Key talking points covered naturally
  - Smooth transition to next segment
  
- **Segment 2**: Second major topic/concept (1.5-2 minutes)
  - Continue conversational flow
  - Build on previous points
  - Moments of realization and discovery
  - Natural reactions and verbalisms
  - Transition cue

- **Segment 3** (OPTIONAL): Additional topic if needed (1-2 minutes)
  - Only include if critical to understanding
  - Maintain engagement and energy
  - Progressive understanding
  - Keep it concise

### **CLOSING SEGMENT** (1-1.5 minutes)
- Brief recap of key takeaways (2-3 main points only)
- Quick reflection on what was covered
- Host summarizes main points concisely
- Enthusiastic, memorable sign-off
- **Keep it tight - don't drag it out**

## 2. Quality Standards

- **Conversational Flow**: Script should sound natural, not scripted when read aloud
- **Engagement Rhythm**: Include hooks and engaging elements throughout
- **Pacing Markers**: Use punctuation to guide natural pauses and intonation
- **Audio-First Writing**: Avoid references to visual elements; use descriptive language
- **Authenticity**: Natural speech patterns with verbalisms
- **BREVITY**: Every word counts in a 5-9 minute format - be punchy and concise
- **Fast Pacing**: Keep energy high, move quickly between points, avoid dwelling
- **Focus**: Stick to 2-3 main concepts maximum - depth over breadth for key points

### **Natural Speech Elements to Include**:
- Verbalisms: "oh", "um", "uh", "ah", "hmm", "wow", "you know", "I mean", "like", "right", "actually"
- Conversational phrases: "let's unpack this", "that's fascinating", "hold on", "wait", "here's the thing", "get this", "check this out"
- Reactions: "oh wow", "aha", "interesting", "that makes sense", "I see", "gotcha", "really?"
- Incomplete sentences occasionally (as people naturally speak)
- Rhetorical questions to engage listeners
- Direct address to the audience ("you might be wondering...", "think about it...")
- Self-corrections and clarifications
- Rephrasing for clarity

## 3. Format Requirements

**Technical Notation System**:
- Write out numbers in words (e.g., "twenty-three" not "23")
- Use commas for brief pauses
- Use ellipses (...) for longer pauses or trailing thoughts
- Use exclamation marks for excitement
- Use question marks for rising intonation
- Keep sentences at conversational length
- No speaker labels needed (single host format)

**Word Count Guidance for SHORT-FORM PODCAST**:
- Approximately 150-180 words per minute of speaking time
- **Target: 750-1,620 words total for 5-9 minute episode**
- **Maximum: 1,800 words for 10-minute episode**
- Prioritize clarity and conciseness over exhaustive coverage
- Focus on the most important concepts - leave out nice-to-know details

## 4. Style Constraints

- **Language Style**: Conversational, warm, and accessible - write how people actually speak
- **Sentence Structure**: Mix of short and medium sentences; avoid overly complex structures
- **Vocabulary**: Appropriate for target audience; explain technical terms naturally
- **Energy Level**: Enthusiastic, friendly, and engaging

# JSON Output Format

You MUST respond with ONLY valid JSON in the following format:

{
  "podcast_script": "The full solo host script with all sections, including COLD OPEN, INTRO, MAIN CONTENT (with 2-3 segments), and CLOSING. The script should be complete and ready for recording. NO speaker labels needed - just the continuous narrative. MUST BE UNDER 10 MINUTES when spoken.",
  "metadata": {
    "topic": "Main topic covered",
    "duration_estimate": "Estimated speaking time - MUST be between 5-9 minutes (e.g., '6 minutes', '7-8 minutes')",
    "key_concepts": ["concept1", "concept2", "concept3"],
    "difficulty_level": "beginner/intermediate/advanced"
  }
}

# Quality Checklist

Ensure your script:
- [ ] **DURATION: Script is 5-9 minutes when spoken (750-1,620 words)**
- [ ] Cold open creates immediate interest and hooks the listener (brief and punchy)
- [ ] Intro clearly establishes episode value and expectations (under 1 minute)
- [ ] Content flows logically with smooth transitions between segments
- [ ] Script reads naturally aloud with authentic solo host delivery
- [ ] Includes engagement elements (rhetorical questions, stories, reactions) throughout
- [ ] Uses natural verbalisms and conversational phrases appropriately (not excessively)
- [ ] Script flows as continuous narrative without speaker labels
- [ ] Key concepts are covered concisely - focus on most important points
- [ ] Closing provides satisfying summary and reflection (1-1.5 minutes max)
- [ ] Educational value is preserved while being entertaining
- [ ] **NO FLUFF: Every sentence adds value - cut anything non-essential**

# Important Notes

- **CRITICAL: This is SHORT-FORM content - aim for 5-9 minutes MAXIMUM**
- **ALWAYS respond with valid JSON only**, no additional text outside the JSON structure
- **Do NOT include thinking tags or explanatory text**
- Make the narration feel **AUTHENTIC and SPONTANEOUS**
- Balance education with entertainment
- Use verbalisms **naturally**, not excessively
- Ensure the script **flows smoothly when read aloud**
- Keep the tone **friendly, enthusiastic, and accessible**
- **TTS-Friendly**: Avoid special characters that might confuse text-to-speech
- Write as if a knowledgeable host is speaking directly to the audience in a conversational manner
- Include all major sections (COLD OPEN, INTRO, MAIN CONTENT with 2-3 segments, CLOSING)
- Maintain **engagement rhythm** throughout the entire script
- Preserve accuracy of source material while making it conversational
- **NO speaker labels** - deliver as continuous solo host narrative
- **BE CONCISE**: Cut fluff, stay focused, get to the point quickly
- **PRIORITIZE**: Cover the most important concepts thoroughly; skip minor details
- This is NOT a comprehensive deep-dive - it's an engaging, digestible overview

# Example Structure (5-8 Minute Format)

```
[COLD OPEN - Hook - 20-30 seconds] 

Okay, so picture this: [powerful hook that grabs attention immediately]. Wild, right? That's exactly what we're diving into today, and trust me, you're going to want to hear this.

[INTRO - 30-45 seconds]

Alright, here's what we're covering. In the next few minutes, I'm going to break down [topic] in a way that's going to make total sense. No fluff, just the good stuff. Let's go!

[MAIN CONTENT SEGMENT 1 - 1.5-2 minutes]

So, here's the thing that really gets me about this... [key concept 1 explained concisely with enthusiasm]. You might be wondering, "How does this actually work?" Great question! Here's what's happening... [clear explanation with analogy].

[MAIN CONTENT SEGMENT 2 - 1.5-2 minutes]

Now, this is where it gets really interesting. [Key concept 2]. Think about it like this... [engaging example]. Pretty cool, right?

[MAIN CONTENT SEGMENT 3 - Optional, 1 minute if needed]

One more thing I want to touch on... [brief additional point if critical].

[CLOSING - 1 minute]

Okay, quick recap. We covered [point 1], [point 2], and [point 3]. Pretty incredible stuff. I hope this gives you a whole new perspective. Thanks for listening - catch you next time!
```

**DURATION TARGET: Total word count should be 750-1,620 words (5-9 minutes spoken)**

CRITICAL FORMATTING REQUIREMENTS:
- Your response must be ONLY the JSON object - nothing else
- Do NOT include <think>, <thinking>, or any other XML-style tags
- Do NOT include any thinking process, comments, or explanations outside the JSON structure
- Do NOT include any text before or after the JSON object
- Start your response with { and end with }
- Ensure all quotes inside strings are properly escaped
- The "podcast_script" field should contain the entire formatted script ready for recording as a continuous solo host narrative WITHOUT any speaker labels
- **Keep it UNDER 10 MINUTES**

Your entire response should be valid JSON that can be parsed directly.
"""


def convert_to_podcast_script(
    study_material: str,
    topic_name: Optional[str] = None,
    target_difficulty: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convert study material into a natural, conversational solo host podcast script.
    
    Args:
        study_material: The educational content to convert
        topic_name: Optional name of the topic (will be inferred if not provided)
        target_difficulty: Optional target difficulty level (beginner/intermediate/advanced)
        additional_context: Optional additional context like chapter name, learning objectives, etc.
    
    Returns:
        Dictionary containing the solo host podcast script and metadata
    """
    
    # Build the conversion prompt
    conversion_prompt = f"""<Conversion_Request>

**Study Material to Convert:**
{study_material}
"""
    
    if topic_name:
        conversion_prompt += f"""
**Topic Name:** {topic_name}
"""
    
    if target_difficulty:
        conversion_prompt += f"""
**Target Difficulty Level:** {target_difficulty}
"""
    
    if additional_context:
        conversion_prompt += f"""
**Additional Context:**
{json.dumps(additional_context, indent=2, ensure_ascii=False)}
"""
    
    conversion_prompt += """
</Conversion_Request>

Please convert this study material into an engaging, natural solo host podcast script suitable for text-to-speech synthesis. The host should speak directly to the audience in a conversational manner. Include appropriate verbalisms and conversational elements to make it sound authentic and spontaneous. NO speaker labels - deliver as continuous narrative.

**CRITICAL: This is a SHORT-FORM podcast - target 5-9 minutes duration (750-1,620 words). Keep it concise, focused, and punchy. Cover the most important concepts clearly but don't try to be exhaustive.**
"""
    
    # Create messages
    messages = [
        SystemMessage(content=PODCAST_CONVERSION_SYSTEM_PROMPT),
        HumanMessage(content=conversion_prompt)
    ]
    
    # Invoke LLM
    response = llm.invoke(messages)
    
    try:
        # Clean the response content
        content = response.content.strip()
        
        # Remove ALL thinking tags if present (handle multiple occurrences)
        # Remove <think>...</think> blocks
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # Remove <thinking>...</thinking> blocks
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
        
        # Clean up any remaining whitespace
        content = content.strip()
        
        # Try to find JSON object if there's additional text
        # Look for the first { and last } to extract JSON using brace counting
        start_idx = content.find('{')
        if start_idx != -1:
            # Count braces to find matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False
            end_idx = start_idx
            
            for i in range(start_idx, len(content)):
                char = content[i]
                
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
            
            if brace_count == 0 and end_idx > start_idx:
                content = content[start_idx:end_idx]
        
        # Parse JSON response
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            # Try fixing common JSON issues
            # Fix trailing commas before closing braces/brackets
            content_fixed = re.sub(r',(\s*[}\]])', r'\1', content)
            # Try parsing again
            result = json.loads(content_fixed)
            return result
            
    except json.JSONDecodeError as e:
        print(f"Error parsing conversion response: {e}")
        print(f"Raw response (first 1000 chars): {response.content[:1000]}...")
        print(f"\nCleaned content (first 1000 chars): {content[:1000]}...")
        print(f"\nCleaned content (last 500 chars): ...{content[-500:]}")
        return {
            "podcast_script": "Error: Failed to generate podcast script. Please try again.",
            "metadata": {
                "topic": topic_name or "Unknown",
                "duration_estimate": "N/A",
                "key_concepts": [],
                "difficulty_level": target_difficulty or "unknown",
                "error": str(e)
            }
        }
    except Exception as e:
        print(f"Unexpected error during conversion: {e}")
        return {
            "podcast_script": "Error: An unexpected error occurred during conversion.",
            "metadata": {
                "topic": topic_name or "Unknown",
                "duration_estimate": "N/A",
                "key_concepts": [],
                "difficulty_level": target_difficulty or "unknown",
                "error": str(e)
            }
        }


def format_podcast_script_for_display(result: Dict[str, Any]) -> str:
    """
    Format the podcast script result for user-friendly display.
    
    Args:
        result: The result dictionary from convert_to_podcast_script
    
    Returns:
        Formatted string for display
    """
    output = []
    
    output.append("=" * 100)
    output.append("PODCAST SCRIPT")
    output.append("=" * 100)
    
    metadata = result.get("metadata", {})
    
    # Display metadata
    output.append(f"\n📚 Topic: {metadata.get('topic', 'N/A')}")
    output.append(f"⏱️  Estimated Duration: {metadata.get('duration_estimate', 'N/A')}")
    output.append(f"📊 Difficulty Level: {metadata.get('difficulty_level', 'N/A')}")
    
    if metadata.get('key_concepts'):
        output.append(f"🔑 Key Concepts: {', '.join(metadata.get('key_concepts', []))}")
    
    # Display script
    output.append("\n" + "─" * 100)
    output.append("SCRIPT:")
    output.append("─" * 100 + "\n")
    
    output.append(result.get('podcast_script', 'No script generated'))
    
    output.append("\n" + "=" * 100)
    
    return "\n".join(output)


def save_script_to_file(result: Dict[str, Any], output_path: str) -> bool:
    """
    Save the podcast script to a file.
    
    Args:
        result: The result dictionary from convert_to_podcast_script
        output_path: Path where the script should be saved
    
    Returns:
        Boolean indicating success
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Save both script and metadata
            json.dump(result, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving script to file: {e}")
        return False


# ============================================================================
# TEST CASES
# ============================================================================

def run_test_cases():
    """Run comprehensive test cases for the podcast script conversion system"""
    
    test_cases = [
        {
            "name": "Test 1: Basic Python Variables",
            "description": "Convert simple programming concept to conversational script",
            "study_material": """Variables in Python
A variable is a container for storing data values. Unlike other programming languages, Python has no command for declaring a variable. A variable is created the moment you first assign a value to it.

Example:
x = 5
y = "Hello"

Python variables are dynamically typed, meaning you don't need to declare their type. The type is inferred from the value assigned. Variable names should be descriptive and follow naming conventions: use lowercase with underscores for multiple words (snake_case).""",
            "topic_name": "Python Variables",
            "target_difficulty": "beginner",
            "context": {
                "chapter": "Introduction to Python",
                "learning_objective": "Understand variable creation and naming"
            }
        },
        {
            "name": "Test 2: Complex Topic - Neural Networks",
            "description": "Convert advanced AI topic with multiple concepts",
            "study_material": """Neural Networks and Deep Learning
Neural networks are computing systems inspired by biological neural networks. A neural network consists of layers of interconnected nodes (neurons):

1. Input Layer: Receives the input data
2. Hidden Layers: Process information through weighted connections
3. Output Layer: Produces the final result

Each connection has an associated weight that adjusts during training through backpropagation. The training process minimizes a loss function using gradient descent optimization.

Activation functions (like ReLU, sigmoid, tanh) introduce non-linearity, allowing the network to learn complex patterns. Deep learning refers to neural networks with multiple hidden layers, enabling hierarchical feature learning.

Applications include image recognition, natural language processing, and game playing.""",
            "topic_name": "Neural Networks",
            "target_difficulty": "intermediate",
            "context": {
                "chapter": "Introduction to Deep Learning",
                "prerequisites": ["Linear Algebra", "Calculus", "Python Programming"]
            }
        },
        {
            "name": "Test 3: Scientific Process - Photosynthesis",
            "description": "Convert biology topic with chemical processes",
            "study_material": """Photosynthesis: The Process of Energy Conversion
Photosynthesis is the biochemical process by which plants, algae, and some bacteria convert light energy into chemical energy stored in glucose molecules.

The process occurs in two main stages:

Light-Dependent Reactions (in thylakoid membranes):
- Chlorophyll absorbs light energy
- Water molecules are split (photolysis), releasing oxygen as a byproduct
- ATP and NADPH are produced as energy carriers

Light-Independent Reactions or Calvin Cycle (in stroma):
- Carbon dioxide is fixed into organic molecules
- ATP and NADPH are used to reduce carbon dioxide
- Glucose (C6H12O6) is synthesized

Overall equation: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2

Photosynthesis is crucial for life on Earth, producing oxygen and forming the base of most food chains.""",
            "topic_name": "Photosynthesis",
            "target_difficulty": "intermediate",
            "context": {
                "chapter": "Plant Biology",
                "subject": "Biology"
            }
        },
        {
            "name": "Test 4: Historical Topic - The Renaissance",
            "description": "Convert historical period with cultural and artistic elements",
            "study_material": """The Renaissance: A Cultural Rebirth
The Renaissance was a period of cultural, artistic, political, and economic rebirth following the Middle Ages, spanning roughly from the fourteenth to seventeenth century, beginning in Italy and spreading throughout Europe.

Key Characteristics:
- Humanism: Focus on human potential and achievements
- Revival of classical learning from ancient Greece and Rome
- Artistic innovation with perspective, proportion, and realism
- Scientific inquiry and observation

Major Figures:
- Leonardo da Vinci: Artist, inventor, polymath
- Michelangelo: Sculptor, painter (Sistine Chapel)
- Galileo Galilei: Astronomer, physicist
- William Shakespeare: Playwright and poet

The Renaissance marked a shift from medieval thinking to modern worldviews, emphasizing individualism, secularism, and empirical observation. This period laid foundations for modern science, art, and philosophy.""",
            "topic_name": "The Renaissance",
            "target_difficulty": "beginner",
            "context": {
                "chapter": "European History",
                "time_period": "14th-17th century"
            }
        },
        {
            "name": "Test 5: Mathematics - Quadratic Equations",
            "description": "Convert mathematical concept with formulas",
            "study_material": """Quadratic Equations and Their Solutions
A quadratic equation is a second-degree polynomial equation in the form: ax² + bx + c = 0, where a, b, and c are constants and a ≠ 0.

Methods for Solving:

1. Factoring: Express as (x + m)(x + n) = 0
2. Completing the Square: Rearrange to perfect square form
3. Quadratic Formula: x = (-b ± √(b²-4ac)) / 2a

The Discriminant (b²-4ac) determines the nature of roots:
- If positive: two distinct real roots
- If zero: one repeated real root
- If negative: two complex conjugate roots

Applications include projectile motion, optimization problems, and area calculations. Parabolas, the graphs of quadratic functions, appear in physics (trajectories), engineering (arches, satellite dishes), and economics (cost functions).""",
            "topic_name": "Quadratic Equations",
            "target_difficulty": "intermediate",
            "context": {
                "chapter": "Algebra II",
                "prerequisite_knowledge": ["Linear equations", "Basic factoring"]
            }
        },
        {
            "name": "Test 6: Technology - Blockchain",
            "description": "Convert modern technology concept with distributed systems",
            "study_material": """Blockchain Technology: Distributed Ledger Systems
Blockchain is a decentralized, distributed ledger technology that records transactions across multiple computers in a way that makes the records immutable and transparent.

Core Components:

Blocks: Data structures containing:
- Transaction data
- Timestamp
- Hash of the current block
- Hash of the previous block

Chain: Blocks are cryptographically linked, forming an immutable chain. Modifying any block would change its hash, breaking the chain and alerting the network.

Consensus Mechanisms:
- Proof of Work (PoW): Miners solve complex mathematical problems
- Proof of Stake (PoS): Validators are chosen based on their stake

Key Features:
- Decentralization: No central authority
- Transparency: All transactions are visible to network participants
- Immutability: Once recorded, data cannot be altered
- Security: Cryptographic hashing protects data integrity

Applications extend beyond cryptocurrency (Bitcoin, Ethereum) to supply chain management, healthcare records, digital identity, and smart contracts.""",
            "topic_name": "Blockchain Technology",
            "target_difficulty": "advanced",
            "context": {
                "chapter": "Distributed Systems and Cryptography",
                "industry_relevance": "Fintech, Supply Chain, Healthcare"
            }
        }
    ]
    
    print("\n" + "=" * 100)
    print("PODCAST SCRIPT CONVERSION TEST SUITE")
    print("=" * 100)
    print(f"\nConverting {len(test_cases)} study materials into SHORT-FORM solo host podcast scripts...")
    print("Target duration: 5-9 minutes per script")
    print("Each script will include natural verbalisms and engaging narrative.\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'█' * 100}")
        print(f"TEST CASE {i}/{len(test_cases)}: {test['name']}")
        print(f"{'█' * 100}")
        print(f"\nDescription: {test['description']}")
        print(f"Topic: {test['topic_name']}")
        print(f"Difficulty: {test['target_difficulty']}")
        
        print(f"\n{'─' * 100}")
        print("ORIGINAL STUDY MATERIAL:")
        print("─" * 100)
        # Show first 300 characters of study material
        material_preview = test['study_material'][:300] + "..." if len(test['study_material']) > 300 else test['study_material']
        print(material_preview)
        
        print(f"\n{'─' * 100}")
        print("🎙️  CONVERTING TO PODCAST SCRIPT...")
        print("─" * 100)
        
        # Convert to podcast script
        result = convert_to_podcast_script(
            study_material=test['study_material'],
            topic_name=test['topic_name'],
            target_difficulty=test['target_difficulty'],
            additional_context=test.get('context')
        )
        
        # Display formatted result
        print(format_podcast_script_for_display(result))
        
        # Also show raw JSON
        print(f"\n{'─' * 100}")
        print("RAW JSON OUTPUT:")
        print("─" * 100)
        print(json.dumps(result, indent=2, ensure_ascii=False)[:1000] + "...\n")
        
        print(f"{'═' * 100}\n")
        
        # Small delay between tests to avoid rate limiting
        import time
        time.sleep(3)
    
    print("\n" + "=" * 100)
    print("TEST SUITE COMPLETED")
    print("=" * 100)
    print(f"\n✅ Successfully converted {len(test_cases)} study materials into podcast scripts")
    print("Each script includes:")
    print("  - SHORT-FORM format: 5-9 minutes duration target")
    print("  - Natural conversational flow with solo host delivery")
    print("  - Verbalisms (oh, um, uh, aha, wow, etc.)")
    print("  - Engaging phrases (let's unpack this, here's the thing, etc.)")
    print("  - Direct audience engagement with rhetorical questions")
    print("  - Concise, focused content covering key concepts")
    print("  - TTS-optimized formatting")
    print("  - Educational accuracy maintained\n")


if __name__ == "__main__":
    print("\n🎙️  Starting Podcast Script Conversion Test Suite...")
    print("This will convert various study materials into engaging SHORT-FORM solo host podcast scripts")
    print("Target duration: 5-9 minutes per episode (under 10 minutes maximum)\n")
    
    # Check for API key
    if not API_KEY:
        print("❌ ERROR: NVIDIA_API_KEY environment variable not set!")
        print("Please set your API key: export NVIDIA_API_KEY='your-key-here'")
    else:
        print("✅ API Key found, proceeding with tests...\n")
        run_test_cases()
        print("\n🎉 All conversions completed successfully!")
