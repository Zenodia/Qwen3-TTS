#!/usr/bin/env python3
"""
Query Qwen3-TTS Gradio API using Python
This script demonstrates how to interact with the Gradio TTS API endpoint.

Usage:
    python query_tts.py "Your text here"
    python query_tts.py --help
"""

import argparse
import json
import requests
import sys
from pathlib import Path


def query_custom_voice(
    text: str,
    host: str = "localhost:8000",
    language: str = "Auto",
    speaker: str = "Vivian",
    instruction: str = "",
    output_file: str = "output.wav"
):
    """
    Query the CustomVoice model endpoint.
    
    Args:
        text: Text to synthesize
        host: Host and port of the Gradio server
        language: Language (Auto, Chinese, English, etc.)
        speaker: Speaker name (Vivian, Serena, Ryan, etc.)
        instruction: Optional instruction for voice control
        output_file: Where to save the audio output
    """
    base_url = f"http://{host}"
    call_url = f"{base_url}/gradio_api/run/run_instruct"
    
    # Prepare the request payload
    payload = {
        "data": [text, language, speaker, instruction]
    }
    
    print(f"Querying Gradio API at {call_url}")
    print(f"Text: {text}")
    print(f"Language: {language}")
    print(f"Speaker: {speaker}")
    if instruction:
        print(f"Instruction: {instruction}")
    print()
    
    try:
        # Step 1: Call the API endpoint
        print("Step 1: Calling API endpoint...")
        response = requests.post(call_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Check if response contains data directly (fast response) or event_id (SSE streaming)
        if "data" in result and result["data"]:
            # Direct response - data is already available
            print("✓ Generation complete (direct response)")
            if isinstance(result["data"], list) and len(result["data"]) > 0:
                audio_data = result["data"][0]
            else:
                audio_data = result["data"]
        elif "event_id" in result:
            # SSE streaming - need to poll for results
            event_id = result["event_id"]
            print(f"Event ID: {event_id}")
            
            # Step 2: Poll for results using Server-Sent Events
            print("Step 2: Waiting for generation to complete...")
            status_url = f"{base_url}/gradio_api/run/run_instruct/{event_id}"
            
            response = requests.get(status_url, stream=True, timeout=120)
            response.raise_for_status()
            
            # Parse SSE stream
            audio_data = None
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        try:
                            data = json.loads(data_str)
                            if isinstance(data, list) and len(data) > 0:
                                audio_data = data[0]
                        except json.JSONDecodeError:
                            continue
            
            if not audio_data:
                print("Error: No audio data received")
                return False
        else:
            print(f"Error: Invalid response format: {result}")
            return False
        
        print("Response received:")
        print(json.dumps(audio_data, indent=2))
        
        # Extract audio file information
        if isinstance(audio_data, dict):
            file_path = audio_data.get("path") or audio_data.get("name")
        elif isinstance(audio_data, str):
            file_path = audio_data
        else:
            print(f"Unexpected audio data format: {audio_data}")
            return False
        
        if not file_path:
            print(f"Error: Could not extract file path from audio data: {audio_data}")
            return False
        
        # Step 3: Download the audio file
        file_url = f"{base_url}/gradio_api/file={file_path}"
        print(f"\nDownloading audio from: {file_url}")
        
        audio_response = requests.get(file_url, timeout=30)
        audio_response.raise_for_status()
        
        # Save the audio file
        with open(output_file, "wb") as f:
            f.write(audio_response.content)
        
        file_size = len(audio_response.content)
        print(f"✓ Success! Audio saved to: {output_file} ({file_size} bytes)")
        return True
            
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to query API: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def query_voice_design(
    text: str,
    host: str = "localhost:8000",
    language: str = "Auto",
    design_instruction: str = "",
    output_file: str = "output.wav"
):
    """
    Query the VoiceDesign model endpoint.
    
    Args:
        text: Text to synthesize
        host: Host and port of the Gradio server
        language: Language (Auto, Chinese, English, etc.)
        design_instruction: Voice design instruction
        output_file: Where to save the audio output
    """
    base_url = f"http://{host}"
    call_url = f"{base_url}/gradio_api/run/run_voice_design"
    
    # Prepare the request payload
    payload = {
        "data": [text, language, design_instruction]
    }
    
    print(f"Querying Gradio API at {call_url}")
    print(f"Text: {text}")
    print(f"Language: {language}")
    print(f"Design: {design_instruction}")
    print()
    
    try:
        # Step 1: Call the API endpoint
        print("Step 1: Calling API endpoint...")
        response = requests.post(call_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Check if response contains data directly (fast response) or event_id (SSE streaming)
        if "data" in result and result["data"]:
            # Direct response - data is already available
            print("✓ Generation complete (direct response)")
            if isinstance(result["data"], list) and len(result["data"]) > 0:
                audio_data = result["data"][0]
            else:
                audio_data = result["data"]
        elif "event_id" in result:
            # SSE streaming - need to poll for results
            event_id = result["event_id"]
            print(f"Event ID: {event_id}")
            
            # Step 2: Poll for results using Server-Sent Events
            print("Step 2: Waiting for generation to complete...")
            status_url = f"{base_url}/gradio_api/run/run_voice_design/{event_id}"
            
            response = requests.get(status_url, stream=True, timeout=120)
            response.raise_for_status()
            
            # Parse SSE stream
            audio_data = None
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        try:
                            data = json.loads(data_str)
                            if isinstance(data, list) and len(data) > 0:
                                audio_data = data[0]
                        except json.JSONDecodeError:
                            continue
            
            if not audio_data:
                print("Error: No audio data received")
                return False
        else:
            print(f"Error: Invalid response format: {result}")
            return False
        
        print("Response received:")
        print(json.dumps(audio_data, indent=2))
        
        # Extract audio file information
        if isinstance(audio_data, dict):
            file_path = audio_data.get("path") or audio_data.get("name")
        elif isinstance(audio_data, str):
            file_path = audio_data
        else:
            print(f"Unexpected audio data format: {audio_data}")
            return False
        
        if not file_path:
            print(f"Error: Could not extract file path from audio data: {audio_data}")
            return False
        
        # Step 3: Download the audio file
        file_url = f"{base_url}/gradio_api/file={file_path}"
        print(f"\nDownloading audio from: {file_url}")
        
        audio_response = requests.get(file_url, timeout=30)
        audio_response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(audio_response.content)
        
        file_size = len(audio_response.content)
        print(f"✓ Success! Audio saved to: {output_file} ({file_size} bytes)")
        return True
            
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to query API: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Query Qwen3-TTS Gradio API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CustomVoice model (default)
  python query_tts.py "Hello world"
  python query_tts.py "Hello" --speaker Ryan --language English
  python query_tts.py "你好" --instruction "用特别愤怒的语气说"
  
  # VoiceDesign model
  python query_tts.py "Hello" --mode voice_design --design "Young female voice"
  
  # Save to specific file
  python query_tts.py "Hello world" --output my_audio.wav
        """
    )
    
    parser.add_argument(
        "text",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--host",
        default="localhost:8000",
        help="Gradio server host:port (default: localhost:8000)"
    )
    parser.add_argument(
        "--mode",
        choices=["custom_voice", "voice_design"],
        default="custom_voice",
        help="Model mode (default: custom_voice)"
    )
    parser.add_argument(
        "--language",
        default="Auto",
        help="Language (default: Auto)"
    )
    parser.add_argument(
        "--speaker",
        default="Vivian",
        help="Speaker name for custom_voice mode (default: Vivian)"
    )
    parser.add_argument(
        "--instruction",
        default="",
        help="Optional instruction for voice control (custom_voice mode)"
    )
    parser.add_argument(
        "--design",
        default="",
        help="Voice design instruction (voice_design mode)"
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output audio file path (default: output.wav)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "custom_voice":
        success = query_custom_voice(
            text=args.text,
            host=args.host,
            language=args.language,
            speaker=args.speaker,
            instruction=args.instruction,
            output_file=args.output
        )
    elif args.mode == "voice_design":
        if not args.design:
            print("Error: --design is required for voice_design mode", file=sys.stderr)
            return 1
        success = query_voice_design(
            text=args.text,
            host=args.host,
            language=args.language,
            design_instruction=args.design,
            output_file=args.output
        )
    else:
        print(f"Error: Unknown mode {args.mode}", file=sys.stderr)
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

