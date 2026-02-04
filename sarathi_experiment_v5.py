

import torch
import json
import re
import os
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import gc

def read_gita_dataset_from_txt(file_path):

    #Reading the Gita dataset from a text file containing JSON-like structure

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Try to parse as JSON first
        try:
            # Clean the content for JSON parsing
            content = content.strip()

            # If content is a list of conversations
            if content.startswith('['):
                data = json.loads(content)

            # If content is a dict with conversations key
            elif content.startswith('{'):
                data = json.loads(content)
                if "conversations" in data:
                    data = data["conversations"]
                elif "data" in data:
                    data = data["data"]
                elif "messages" in data:
                    data = data["messages"]

            # Process the loaded data
            conversations = []

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, list) and len(item) >= 3:
                        # Ensure each message has the right structure
                        conv = []
                        for msg in item[:3]:
                            if isinstance(msg, dict):
                                conv.append(msg)
                            else:
                                # Try to convert if not a dict
                                conv.append({"role": "unknown", "content": str(msg)})
                        if len(conv) == 3:
                            conversations.append(conv)
                    elif isinstance(item, dict):
                        # Check if it's a single conversation with messages
                        if "messages" in item:
                            messages = item["messages"]
                            if len(messages) >= 3:
                                conversations.append(messages[:3])
                        elif "conversation" in item:
                            conversation = item["conversation"]
                            if isinstance(conversation, list) and len(conversation) >= 3:
                                conversations.append(conversation[:3])

            return conversations

        except json.JSONDecodeError:
            # Fallback to regex extraction
            return extract_conversations_from_text(content)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def extract_conversations_from_text(content):
    """
    Extract conversation data from text format using regex
    """
    conversations = []

    # Try to find JSON arrays in the content
    # Look for patterns like: [{...}, {...}, {...}]
    json_pattern = r'\[\s*\{[^}]+\}\s*(?:,\s*\{[^}]+\}\s*){2,}\]'

    matches = re.findall(json_pattern, content, re.DOTALL)

    for match in matches:
        try:
            # Clean the match for JSON parsing
            clean_match = re.sub(r'[\n\r]+', ' ', match)
            clean_match = re.sub(r'\s+', ' ', clean_match)

            # Parse as JSON
            conv_data = json.loads(clean_match)
            if isinstance(conv_data, list) and len(conv_data) >= 3:
                conversations.append(conv_data[:3])
        except:
            continue

    # If no JSON found, try to extract structured text
    if not conversations:
        conversations = extract_from_structured_text(content)

    return conversations

def extract_from_structured_text(content):
    """Extract conversations from structured text format"""
    conversations = []

    # Split by potential conversation separators
    sections = re.split(r'\n\s*\n|\r\n\s*\r\n', content)

    for section in sections:
        lines = section.strip().split('\n')
        if len(lines) >= 3:
            conv = []
            for line in lines[:3]:
                # Try to parse as JSON
                if line.strip().startswith('{'):
                    try:
                        msg = json.loads(line.strip())
                        conv.append(msg)
                    except:
                        # If not JSON, create a message from text
                        conv.append({
                            "role": "unknown",
                            "content": line.strip()
                        })
                else:
                    # Create message from text
                    conv.append({
                        "role": "unknown",
                        "content": line.strip()
                    })

            if len(conv) == 3:
                conversations.append(conv)

    return conversations

def create_training_dataset_from_conversations(conversations):
    """
    Convert conversations to training dataset format
    """
    formatted_data = []

    print(f"Processing {len(conversations)} conversations...")

    for i, conv in enumerate(conversations):
        try:
            if len(conv) >= 3:
                system_msg, user_msg, assistant_msg = conv[0], conv[1], conv[2]

                # Extract text content safely
                def extract_text(msg):
                    if not isinstance(msg, dict):
                        return str(msg)

                    # Try different possible field names
                    text_fields = ["content", "text", "value", "message", "response"]

                    for field in text_fields:
                        if field in msg:
                            content = msg[field]
                            if isinstance(content, str):
                                return content
                            elif isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and "text" in item:
                                        return item.get("text", "")
                                    elif isinstance(item, str):
                                        return item
                            elif isinstance(content, dict) and "text" in content:
                                return content.get("text", "")

                    # If no text found, try to convert the whole message
                    return json.dumps(msg)

                system_text = extract_text(system_msg)
                user_text = extract_text(user_msg)
                assistant_text = extract_text(assistant_msg)

                # Clean up texts
                system_text = system_text.strip()
                user_text = user_text.strip()
                assistant_text = assistant_text.strip()

                # Skip empty conversations
                if system_text and user_text and assistant_text:
                    formatted_data.append({
                        "instruction": system_text,
                        "input": user_text,
                        "output": assistant_text
                    })

                    # Print first 3 examples
                    if i < 3:
                        print(f"\nExample {i+1}:")
                        print(f"  Instruction: {system_text[:100]}...")
                        print(f"  Input: {user_text[:100]}...")
                        print(f"  Output: {assistant_text[:100]}...")

        except Exception as e:
            print(f"Error processing conversation {i+1}: {e}")
            continue

    print(f"\nSuccessfully processed {len(formatted_data)} training examples")

    if len(formatted_data) == 0:
        print("WARNING: No valid training examples found!")
        # Create dummy dataset for testing
        formatted_data = [{
            "instruction": "You are Krishna, guide from Bhagavad Gita.",
            "input": "What is the purpose of life?",
            "output": "The purpose of life is to fulfill your dharma (duty) with devotion."
        }]

    return Dataset.from_list(formatted_data)

def save_model_in_2bit(model, tokenizer, save_path):
    """
    Save the model in 2-bit quantization format
    """
    print(f"\nSaving model to {save_path}...")

    # Save tokenizer first
    tokenizer.save_pretrained(save_path)
    print("✓ Tokenizer saved")

    # For Unsloth models, use their save method
    try:
        # Save the merged model (base + LoRA) in 16-bit for better compatibility
        model.save_pretrained_merged(
            save_path,
            tokenizer,
            save_method="merged_16bit",
        )
        print("✓ Model saved in 16-bit (merged with LoRA)")

        # Now load and save in 2-bit
        print("Converting to 2-bit format...")
        quantization_config = BitsAndBytesConfig(
            load_in_2bit=True,
            bnb_2bit_quant_type="nf4",
            bnb_2bit_compute_dtype=torch.float16,
            bnb_2bit_use_double_quant=True,
        )

        # Load the saved model in 2-bit
        model_2bit = AutoModelForCausalLM.from_pretrained(
            save_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # Save again in 2-bit
        model_2bit.save_pretrained(
            save_path,
            safe_serialization=True
        )

        # Save quantization config
        with open(os.path.join(save_path, "quantization_config.json"), "w") as f:
            json.dump(quantization_config.to_dict(), f, indent=2)

        print("✓ Model saved in 2-bit format")

        # Clean up
        del model_2bit
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Error saving in 2-bit: {e}")
        print("Saving in 4-bit instead...")

        # Fallback to 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model.save_pretrained_merged(
            save_path,
            tokenizer,
            save_method="merged_4bit",
            tokenizer_padding_side="right",
        )

        # Save quantization config
        with open(os.path.join(save_path, "quantization_config.json"), "w") as f:
            json.dump(quantization_config.to_dict(), f, indent=2)

        print("✓ Model saved in 4-bit format")

    print(f"\nModel successfully saved to {save_path}")

def save_model_fp16(model, tokenizer, save_path):
    """
    Save merged LoRA + base model in FP16 (GGUF-compatible)
    """
    print(f"\nSaving FP16 merged model to {save_path}...")

    os.makedirs(save_path, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    print("✓ Tokenizer saved")

    # Save merged FP16 model
    model.save_pretrained_merged(
        save_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print("✓ Model saved in FP16 (merged, GGUF-ready)")

    torch.cuda.empty_cache()
    gc.collect()

def main():
    # Configuration
    max_seq_length = 2048
    output_dir = "krishna-counselor-fp16"

    # File path - adjust as needed
    file_path = "gita_dataset_train_example_v1.txt"

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        print("Please make sure the file is in the current directory or provide the full path.")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", os.listdir('.'))
        return

    print(f"Reading dataset from: {file_path}")

    # Read and process the dataset
    conversations = read_gita_dataset_from_txt(file_path)

    if not conversations:
        print("No conversations found in the file. Check the file format.")
        print("Creating sample dataset for testing...")
        # Create a sample conversation
        conversations = [[
            {"role": "system", "content": "You are Krishna from Bhagavad Gita."},
            {"role": "user", "content": "How to deal with stress?"},
            {"role": "assistant", "content": "Perform your duty without attachment to results."}
        ]]

    print(f"Found {len(conversations)} conversations")

    # Create training dataset
    dataset = create_training_dataset_from_conversations(conversations)

    print(f"\nDataset created with {len(dataset)} examples")
    print(f"Dataset features: {dataset.column_names}")

    # Show sample
    if len(dataset) > 0:
        print("\nSample from dataset:")
        print(f"Instruction: {dataset[0]['instruction'][:100]}...")
        print(f"Input: {dataset[0]['input'][:100]}...")
        print(f"Output: {dataset[0]['output'][:100]}...")

    # MODEL LOADING in 2-bit
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)

    try:
        # Load with 2-bit quantization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-3B-Instruct",
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_2bit=True,
        )
        print("✓ Model loaded in 2-bit quantization")

    except Exception as e:
        print(f"Error loading in 2-bit: {e}")
        print("Falling back to 4-bit...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-3B-Instruct",
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        print("✓ Model loaded in 4-bit quantization")

    EOS_TOKEN = tokenizer.eos_token
    if not EOS_TOKEN:
        EOS_TOKEN = "<|endoftext|>"
        tokenizer.eos_token = EOS_TOKEN

    # APPLY LORA for targeted training
    print("\nApplying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=3407,
    )
    print("✓ LoRA applied")

    # DATA FORMATTING - FIXED VERSION
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    def formatting_prompts_func(examples):
        """Format examples for training. Must return a list of strings."""
        texts = []

        # Get the values
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        # Ensure we're working with lists
        if not isinstance(instructions, list):
            instructions = [instructions]
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]

        # Format each example
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            # Skip empty examples
            if not instruction or not input_text or not output:
                continue

            text = alpaca_prompt.format(
                instruction,
                input_text,
                output
            ) + EOS_TOKEN
            texts.append(text)

        return texts  # Return list directly, not dictionary

    # TEST THE FORMATTING FUNCTION
    print("\nTesting formatting function...")
    test_example = {
        "instruction": ["You are Krishna."],
        "input": ["How to be happy?"],
        "output": ["Do your duty without attachment."]
    }

    formatted = formatting_prompts_func(test_example)
    print(f"Formatted text sample (first 200 chars): {formatted[0][:200]}...")
    print("✓ Formatting function works correctly")

    # TRAINING SETUP
    print("\n" + "="*60)
    print("SETTING UP TRAINER")
    print("="*60)

    # Disable packing for debugging
    packing = False  # Set to False for debugging

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=max_seq_length,
        dataset_text_field="text",  # Not used when formatting_func is provided
        formatting_func=formatting_prompts_func,  # This should return list of strings
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="training_output",
            report_to="none",
            save_strategy="epoch",
            save_total_limit=1,
            ddp_find_unused_parameters=False,
        ),
    )

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    # Train
    trainer_stats = trainer.train()

    print("\nTraining completed!")
    print(f"Training stats: {trainer_stats}")

    # TEST THE MODEL
    print("\n" + "="*60)
    print("TESTING TRAINED MODEL")
    print("="*60)

    FastLanguageModel.for_inference(model)

    test_prompt = alpaca_prompt.format(
        "You are Krishna, a wise guide who provides advice based on the Bhagavad Gita.",
        "I feel overwhelmed by my responsibilities at work.",
        ""
    )

    inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)

    print("Test input: 'I feel overwhelmed by my responsibilities at work.'")
    print("\nModel response:")

    outputs = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*60)
    print("FULL RESPONSE:")
    print("="*60)
    print(response)

    # SAVE THE MODEL
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)

    # Save using custom function
    save_model_fp16(model, tokenizer, output_dir)

    # Save training examples for reference
    sample_data = []
    for i in range(min(5, len(dataset))):
        sample_data.append({
            "instruction": dataset[i]["instruction"],
            "input": dataset[i]["input"],
            "output": dataset[i]["output"]
        })

    with open(f"{output_dir}/training_samples.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    print("✓ Sample training data saved")

    # Also save the training arguments
    trainer.save_model(output_dir + "_trainer")
    print("✓ Trainer state saved")

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Model: Qwen2.5-3B-Instruct")
    print(f"Quantization: 2-bit (or 4-bit fallback)")
    print(f"Training examples: {len(dataset)}")
    print(f"Epochs: 3")
    print(f"Model saved to: {output_dir}/")
    print(f"\nTo load the model:")
    print(f"from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig")
    print(f"import torch")
    print(f"\n# For 2-bit loading:")
    print(f"quantization_config = BitsAndBytesConfig(")
    print(f"    load_in_2bit=True,")
    print(f"    bnb_2bit_quant_type='nf4',")
    print(f"    bnb_2bit_compute_dtype=torch.float16,")
    print(f"    bnb_2bit_use_double_quant=True,")
    print(f")")
    print(f"model = AutoModelForCausalLM.from_pretrained('{output_dir}',")
    print(f"    quantization_config=quantization_config,")
    print(f"    device_map='auto',")
    print(f"    trust_remote_code=True)")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")

if __name__ == "__main__":
    main()

import torch
import json
import re
import os
import subprocess
import shutil
from datetime import datetime
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import gc


def convert_to_gguf(model_path, output_name="krishna-counselor", quant_type="q4_k_m"):
    """
    Convert a trained model to GGUF format for mobile deployment
    """
    print(f"\n" + "="*60)
    print("CONVERTING TO GGUF FORMAT")
    print("="*60)

    # Create output directory
    gguf_dir = "gguf_models"
    os.makedirs(gguf_dir, exist_ok=True)

    # Step 1: Clone llama.cpp if not exists
    print("1. Setting up llama.cpp...")
    if not os.path.exists("llama.cpp"):
        try:
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
                          check=True, capture_output=True, text=True)
            print("✓ llama.cpp cloned")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to clone llama.cpp: {e}")
            return None
    else:
        print("✓ llama.cpp already exists")

    # Step 2: Build llama.cpp tools
    print("\n2. Building llama.cpp tools...")
    os.chdir("llama.cpp")

    # Build convert and quantize tools
    # try:
    #     # Clean and build
    #     subprocess.run(["make", "clean"], capture_output=True, text=True)
    #     subprocess.run(["make", "-j4"], check=True, capture_output=True, text=True)
    #     print("✓ llama.cpp tools built")
    # except subprocess.CalledProcessError as e:
    #     print(f"✗ Failed to build tools: {e}")
    #     os.chdir(".")
    #     return None

    # Step 3: Convert to GGUF (FP16)
    print(f"\n3. Converting model to GGUF format...")
    fp16_gguf = f"../{gguf_dir}/{output_name}-f16.gguf"

    convert_cmd = [
        "python", "./convert.py",
        f"../{model_path}",
        "--outtype", "f16",
        "--outfile", fp16_gguf,
        "--vocab-type", "bpe"
    ]

    print(f"Running: {' '.join(convert_cmd)}")
    try:
        result = subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        print("✓ Conversion successful")
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        os.chdir(".")
        return None

    # Step 4: Quantize the GGUF file
    print(f"\n4. Quantizing to {quant_type}...")
    quant_gguf = f"../{gguf_dir}/{output_name}-{quant_type}.gguf"

    if os.path.exists("./quantize"):
        quantize_cmd = [
            "./quantize",
            fp16_gguf,
            quant_gguf,
            quant_type
        ]

        print(f"Running: {' '.join(quantize_cmd)}")
        try:
            result = subprocess.run(quantize_cmd, check=True, capture_output=True, text=True)
            print("✓ Quantization successful")
        except subprocess.CalledProcessError as e:
            print(f"✗ Quantization failed: {e}")
            # Use FP16 as fallback
            quant_gguf = fp16_gguf
            print(f"Using FP16 as fallback: {quant_gguf}")
    else:
        print("✗ quantize tool not found, using FP16 only")
        quant_gguf = fp16_gguf

    # Go back to original directory
    os.chdir(".")

    # Step 5: Verify and get file info
    if os.path.exists(quant_gguf):
        file_size = os.path.getsize(quant_gguf) / 1024 / 1024  # MB
        print(f"\n5. GGUF file created successfully!")
        print(f"   File: {quant_gguf}")
        print(f"   Size: {file_size:.1f} MB")
        print(f"   Quantization: {quant_type}")

        # Create additional quantization formats for mobile
        create_additional_quantizations(quant_gguf, gguf_dir, output_name)

        return quant_gguf
    else:
        print("✗ GGUF file creation failed")
        return None

def create_additional_quantizations(base_gguf, gguf_dir, output_name):
    """Create additional quantization formats for mobile optimization"""
    print(f"\n6. Creating additional quantization formats...")

    if not os.path.exists(base_gguf):
        print("✗ Base GGUF not found, skipping additional quantization")
        return

    # Common quantization types for mobile
    mobile_quants = [
        ("q4_k_m", "Good quality, moderate size"),
        ("q3_k_m", "Balanced quality/size"),
        ("q2_k", "Smallest, basic quality"),
        ("q5_k_m", "Better quality, larger size"),
    ]

    created_files = []

    for quant_type, description in mobile_quants:
        if quant_type in base_gguf:  # Skip if already this type
            continue

        output_file = os.path.join(gguf_dir, f"{output_name}-{quant_type}.gguf")

        if os.path.exists("llama.cpp/quantize"):
            try:
                subprocess.run([
                    "llama.cpp/quantize",
                    base_gguf,
                    output_file,
                    quant_type
                ], check=True, capture_output=True, text=True)

                if os.path.exists(output_file):
                    size_mb = os.path.getsize(output_file) / 1024 / 1024
                    created_files.append({
                        "file": output_file,
                        "quant": quant_type,
                        "size_mb": size_mb,
                        "description": description
                    })
                    print(f"  ✓ {quant_type}: {size_mb:.1f} MB")
            except:
                pass  # Skip if quantization fails

    return created_files

def create_mobile_deployment_package(gguf_file, output_name="krishna-counselor"):
    """
    Create a complete mobile deployment package
    """
    print(f"\n" + "="*60)
    print("CREATING MOBILE DEPLOYMENT PACKAGE")
    print("="*60)

    # Create deployment directory
    deploy_dir = "mobile_deployment"
    os.makedirs(deploy_dir, exist_ok=True)

    # Copy GGUF files
    gguf_dir = "gguf_models"
    if os.path.exists(gguf_dir):
        for file in os.listdir(gguf_dir):
            if file.endswith(".gguf"):
                src = os.path.join(gguf_dir, file)
                dst = os.path.join(deploy_dir, file)
                shutil.copy2(src, dst)
                print(f"✓ Copied: {file}")

    # Create model card
    model_card = f"""# Krishna Counselor - Mobile Model

## Model Description
This is a fine-tuned version of Qwen2.5-3B-Instruct trained to act as Krishna from the Bhagavad Gita, providing spiritual guidance and advice.

## Model Files
Available in GGUF format with different quantizations:
- **q4_k_m**: Best balance of quality and size (recommended for most devices)
- **q3_k_m**: Good quality with smaller size
- **q2_k**: Smallest size, basic quality
- **q5_k_m**: Higher quality, larger size
- **f16**: Full precision (largest)

## Prompt Format
Use the following format for best results:

```
### Instruction:
{{Your instruction here}}

### Input:
{{Your input or context here}}

### Response:
{{The model's response will go here}}
```

Example:

```
### Instruction:
You are Krishna, a wise guide who provides advice based on the Bhagavad Gita.

### Input:
I feel overwhelmed by my responsibilities at work.

### Response:
```

## Usage Instructions

This model can be loaded and used with `llama.cpp` compatible inference engines.

For example, using the `main` executable from `llama.cpp`:

```bash
./main -m {output_name}-q4_k_m.gguf -p "### Instruction:\nYou are Krishna, a wise guide.\n\n### Input:\nHow can I find peace?\n\n### Response:"
```

For mobile applications, integrate with a GGUF compatible library (e.g., Llama.cpp for iOS/Android).

## Model Details
- **Base Model**: Qwen2.5-3B-Instruct
- **Fine-tuning Dataset**: Custom Gita dataset
- **Quantization**: Mixed GGUF quantizations
- **License**: Refer to the base model's license.

"""

    with open(os.path.join(deploy_dir, "MODEL_CARD.md"), "w") as f:
        f.write(model_card)
    print("✓ MODEL_CARD.md created")

    # Optionally create a simple inference script example
    inference_script_content = f"""# Example Inference Script

#!/bin/bash

MODEL_NAME="{output_name}-q4_k_m.gguf"
PROMPT="### Instruction:\nYou are Krishna, a wise guide.\n\n### Input:\nHow can I deal with anger?\n\n### Response:"

if [ ! -f "$MODEL_NAME" ]; then
    echo "Error: Model file $MODEL_NAME not found. Please ensure it's in the same directory."
    exit 1
fi

if [ ! -f "./llama.cpp/main" ]; then
    echo "Error: llama.cpp/main executable not found. Please build llama.cpp first."
    echo "cd llama.cpp && make && cd .."
    exit 1
fi

./llama.cpp/main -m "$MODEL_NAME" -p "$PROMPT" -n 128 --temp 0.7 --top-k 40 --top-p 0.9 --repeat-penalty 1.1
"""
    with open(os.path.join(deploy_dir, "run_inference.sh"), "w") as f:
        f.write(inference_script_content)
    os.chmod(os.path.join(deploy_dir, "run_inference.sh"), 0o755)
    print("✓ run_inference.sh created")

    print(f"\nMobile deployment package created in: {deploy_dir}/")
    print("You can now download the `mobile_deployment` folder.")

!cd ..

# Download llama binary
!wget https://github.com/ggml-org/llama.cpp/releases/download/b7502/llama-b7502-bin-ubuntu-x64.tar.gz

!mkdir llama-bin
!tar -xvzf llama-b7502-bin-ubuntu-x64.tar.gz -C llama-bin

# Convert model to GGUF format
!python convert_hf_to_gguf.py /content/krishna-counselor-fp16 --outtype f16 --outfile gguf_models/krishna-counselor-f16.gguf

# Quantize to different levels
!llama-bin/llama-quantize gguf_models/krishna-counselor-f16.gguf gguf_models/krishna-counselor-q2_k.gguf q2_k
!llama-bin/llama-quantize gguf_models/krishna-counselor-f16.gguf gguf_models/krishna-counselor-q3_k_m.gguf q3_k_m
!llama-bin/llama-quantize gguf_models/krishna-counselor-f16.gguf gguf_models/krishna-counselor-q4_k_m.gguf q4_k_m

from google.colab import drive
drive.mount('/content/drive')

!cp -r gguf_models/ drive/MyDrive/SAARATHI/

from huggingface_hub import HfFolder

# @title Hugging Face Login (required to upload)
# You need to provide your Hugging Face token here.
# Go to https://huggingface.co/settings/tokens to create a new token.
# Ensure it has 'write' access.

hf_token = "hf_YOUR_TOKEN_HERE" # @param {type: "string"}

# Save your token (optional, but good practice for repeated use)
# HfFolder.save_token(hf_token)

# Define your Hugging Face repository ID
# Replace 'your-username' with your Hugging Face username
# Replace 'krishna-counselor-model' with your desired model name on Hugging Face
repo_id = "your-username/krishna-counselor-model" # @param {type: "string"}

# Push the model to Hugging Face Hub
# The output_dir from previous steps should be 'krishna-counselor-2bit'
output_dir = "krishna-counselor-2bit"

# Reload the model and tokenizer to ensure they are in the correct format for pushing
# We'll load the 2-bit model for upload.

print("Loading model for upload...")
model_to_upload, tokenizer_to_upload = FastLanguageModel.from_pretrained(
    output_dir,
    load_in_2bit=True, # Assuming 2-bit was successful, else try load_in_4bit=True
    torch_dtype=None, # Auto-detects based on load_in_Xbit
)

print(f"Pushing model to Hugging Face Hub: {repo_id}")
model_to_upload.push_to_hub(repo_id, token=hf_token)
tokenizer_to_upload.push_to_hub(repo_id, token=hf_token)

print(f"Model and tokenizer successfully pushed to https://huggingface.co/{repo_id}")

