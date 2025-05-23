import os
import json
import re
import csv

# Set the base directory
BASE_DIR = "/Users/casey/Documents/GitHub/LLM_Healthcare"
OUTPUT_DIR = os.path.join(BASE_DIR, "output/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_data():
    """Parse the clinical text data"""
    print("Parsing the clinical text data...")
    
    # Try to read from paste.txt file
    paste_path = os.path.join(BASE_DIR, "paste.txt")
    if os.path.exists(paste_path):
        try:
            with open(paste_path, 'r') as f:
                content = f.read()
            
            # Split the content by the delimiter
            lines = content.split('\n')
            header = lines[0].split('\t')
            
            data = []
            for i, line in enumerate(lines[1:]):
                if not line.strip():
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 1:
                    record = {
                        "id": f"record_{i}",
                        "text": parts[0],
                        "label": parts[1] if len(parts) > 1 else "0"
                    }
                    data.append(record)
            
            print(f"Successfully loaded {len(data)} records from paste.txt")
            return data
        
        except Exception as e:
            print(f"Error reading paste.txt: {e}")
    
    # If paste.txt failed or doesn't exist, try clean_admission.csv
    try:
        data = []
        csv_path = os.path.join(BASE_DIR, "clean_admission.csv")
        
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            return []
            
        with open(csv_path, 'r') as f:
            content = f.read().strip()
            if not content:
                print("File is empty")
                return []
                
            f.seek(0)  # Reset file pointer
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)  # Skip header if exists
            
            if not header:
                print("File has no content or header")
                return []
                
            for i, row in enumerate(reader):
                if len(row) >= 1:
                    record = {
                        "id": f"record_{i}",
                        "text": row[0],
                        "label": row[1] if len(row) > 1 else "0"
                    }
                    data.append(record)
                    
        print(f"Successfully loaded {len(data)} records from clean_admission.csv")
        return data
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def extract_medications(text):
    """Extract medications from the clinical text using improved patterns"""
    medications = []
    
    # Pattern for DISCHARGE MEDICATIONS section
    med_section_match = re.search(r'(?:DISCHARGE MEDICATIONS?:|Discharge Medications?:)(.*?)(?:DISPOSITION:|DISCHARGE DIAGNOSIS:|DISCHARGE CONDITION:|DISCHARGE INSTRUCTIONS:|FOLLOWUP|\Z)', text, re.DOTALL | re.IGNORECASE)
    
    if med_section_match:
        med_section = med_section_match.group(1).strip()
        print(f"Found medication section: {med_section[:100]}...")
        
        # Pattern 1: Look for numbered medications (e.g., "1. Medication Name XX mg...")
        numbered_meds = re.findall(r'(?:\d+\.\s*)([A-Za-z][A-Za-z0-9\s-]+)\s+(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|mEq|unit|%)', med_section)
        
        # Pattern 2: Look for comma-separated medications (e.g., "Medication Name XX mg, ...")
        comma_meds = re.findall(r'([A-Za-z][A-Za-z0-9\s-]+)\s+(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|mEq|unit|%)[^,]*(?=,|\.|$)', med_section)
        
        # Combine results and deduplicate
        all_meds = list(set(numbered_meds + comma_meds))
        
        for med_name, dosage in all_meds:
            medications.append({
                "name": med_name.strip(),
                "dosage": dosage.strip()
            })
            print(f"Found medication: {med_name.strip()} {dosage.strip()} mg/mcg")
    
    return medications

def extract_diagnoses(text):
    """Extract diagnoses from the clinical text using improved patterns"""
    diagnoses = []
    
    # Pattern for DISCHARGE DIAGNOSIS section
    diag_section_match = re.search(r'(?:DISCHARGE DIAGNOSIS:|Discharge Diagnosis:)(.*?)(?:DISCHARGE CONDITION:|DISCHARGE INSTRUCTIONS:|DISCHARGE MEDICATIONS:|FOLLOWUP|\Z)', text, re.DOTALL | re.IGNORECASE)
    
    if diag_section_match:
        diag_section = diag_section_match.group(1).strip()
        print(f"Found diagnosis section: {diag_section[:100]}...")
        
        # Pattern 1: Look for numbered diagnoses (e.g., "1. Diagnosis Name")
        numbered_diags = re.findall(r'(?:\d+\.\s*)([A-Za-z][A-Za-z0-9\s,-]+?)(?=\d+\.|\n\n|\n[A-Z]|\Z)', diag_section)
        
        # If no numbered diagnoses found, try to extract by lines
        if not numbered_diags:
            # Split by lines and filter out empty lines
            diag_lines = [line.strip() for line in diag_section.split('\n') if line.strip()]
            for line in diag_lines:
                if line and not line.startswith("DISCHARGE") and not line.startswith("Discharge"):
                    diagnoses.append(line)
                    print(f"Found diagnosis: {line}")
        else:
            for diag in numbered_diags:
                diagnoses.append(diag.strip())
                print(f"Found diagnosis: {diag.strip()}")
    
    return diagnoses

def extract_entities(records):
    """Extract medications and diagnoses from each record"""
    for record in records:
        text = record["text"]
        
        # Extract medications
        medications = extract_medications(text)
        record["medications"] = medications
        
        # Extract diagnoses
        diagnoses = extract_diagnoses(text)
        record["diagnoses"] = diagnoses
        
        print(f"\nRecord {record['id']}:")
        print(f"  Found {len(medications)} medications and {len(diagnoses)} diagnoses")
    
    return records

def read_templates():
    """Read templates from the templates-all.csv file"""
    templates_path = os.path.join(BASE_DIR, "templates-all.csv")
    
    if not os.path.exists(templates_path):
        print(f"Templates file not found at {templates_path}")
        return []
    
    try:
        templates = []
        with open(templates_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            for row in reader:
                if len(row) >= 2:
                    template_type = row[0].lower()
                    question = row[1]
                    
                    templates.append({
                        "type": template_type,
                        "question": question
                    })
        
        print(f"Loaded {len(templates)} templates from {templates_path}")
        # Print a few examples
        for i, template in enumerate(templates[:5]):
            print(f"  Template {i+1}: {template['type']} - {template['question']}")
        
        return templates
    
    except Exception as e:
        print(f"Error reading templates: {e}")
        
        # Create simple default templates
        templates = [
            {"type": "medication", "question": "What is the dosage of [MEDICATION]?"},
            {"type": "medication", "question": "Is [MEDICATION] prescribed for this patient?"},
            {"type": "diagnosis", "question": "Does the patient have [DIAGNOSIS]?"},
            {"type": "diagnosis", "question": "What are the patient's diagnoses?"}
        ]
        
        print(f"Using {len(templates)} default templates")
        return templates

def generate_qa_pairs(records, templates):
    """Generate QA pairs from the records and templates"""
    qa_pairs = []
    
    for record in records:
        text = record["text"]
        medications = record.get("medications", [])
        diagnoses = record.get("diagnoses", [])
        
        # Process medication templates
        for template in templates:
            if "medication" in template["type"].lower() and "[MEDICATION]" in template["question"]:
                for med in medications:
                    med_name = med.get("name", "")
                    if not med_name:
                        continue
                    
                    # Create question by replacing placeholder
                    question = template["question"].replace("[MEDICATION]", med_name)
                    
                    # Generate appropriate answer
                    if "dosage" in question.lower():
                        answer = f"{med.get('dosage', '')} mg"
                    elif "prescribed" in question.lower() or "is" in question.lower():
                        answer = "Yes"
                    else:
                        answer = f"{med_name} {med.get('dosage', '')} mg"
                    
                    qa_pairs.append({
                        "id": f"{record['id']}_med_{len(qa_pairs)}",
                        "question": question,
                        "answer": answer,
                        "context": text
                    })
            
            # Process diagnosis templates
            elif "diagnosis" in template["type"].lower() and "[DIAGNOSIS]" in template["question"]:
                for diag in diagnoses:
                    if not diag:
                        continue
                    
                    # Create question by replacing placeholder
                    question = template["question"].replace("[DIAGNOSIS]", diag)
                    
                    # For diagnosis questions, the answer is usually "Yes"
                    answer = "Yes"
                    
                    qa_pairs.append({
                        "id": f"{record['id']}_diag_{len(qa_pairs)}",
                        "question": question,
                        "answer": answer,
                        "context": text
                    })
            
            # Process general diagnosis templates (without placeholders)
            elif "diagnosis" in template["type"].lower() and "[DIAGNOSIS]" not in template["question"]:
                if diagnoses:
                    question = template["question"]
                    answer = "; ".join(diagnoses)
                    
                    qa_pairs.append({
                        "id": f"{record['id']}_diag_list_{len(qa_pairs)}",
                        "question": question,
                        "answer": answer,
                        "context": text
                    })
    
    print(f"Generated {len(qa_pairs)} QA pairs")
    return qa_pairs

def create_squad_format(qa_pairs):
    """Convert QA pairs to SQuAD format"""
    print("Creating SQuAD format...")
    
    squad_data = {
        "version": "v2.0",
        "data": []
    }
    
    # Group by context
    context_groups = {}
    for qa in qa_pairs:
        context = qa["context"]
        if context not in context_groups:
            context_groups[context] = []
        context_groups[context].append(qa)
    
    for context, qas in context_groups.items():
        article = {
            "title": "Medical Record",
            "paragraphs": [
                {
                    "context": context,
                    "qas": []
                }
            ]
        }
        
        for qa in qas:
            # Find the position of the answer in the context
            answer_text = qa["answer"]
            answer_start = context.find(answer_text)
            
            # If exact match not found, use 0 as the position
            if answer_start == -1:
                answer_start = 0
            
            squad_qa = {
                "id": qa["id"],
                "question": qa["question"],
                "answers": [
                    {
                        "text": answer_text,
                        "answer_start": answer_start
                    }
                ],
                "is_impossible": False
            }
            
            article["paragraphs"][0]["qas"].append(squad_qa)
        
        squad_data["data"].append(article)
    
    return squad_data

def main():
    # Parse data
    records = parse_data()
    
    if not records:
        print("No records to process. Exiting.")
        return
    
    # Extract entities 
    enriched_records = extract_entities(records)
    
    # Save the enriched records to JSON
    records_path = os.path.join(OUTPUT_DIR, "patient_records.json")
    with open(records_path, "w") as f:
        json.dump(enriched_records, f, indent=2)
    print(f"Saved {len(enriched_records)} patient records to {records_path}")
    
    # Read templates
    templates = read_templates()
    
    # Generate QA pairs
    qa_pairs = generate_qa_pairs(enriched_records, templates)
    
    # Save QA pairs to JSON
    qa_path = os.path.join(OUTPUT_DIR, "qa_pairs.json")
    with open(qa_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"Saved {len(qa_pairs)} QA pairs to {qa_path}")
    
    # Create and save SQuAD format
    squad_data = create_squad_format(qa_pairs)
    squad_path = os.path.join(OUTPUT_DIR, "squad_format.json")
    with open(squad_path, "w") as f:
        json.dump(squad_data, f, indent=2)
    print(f"Created SQuAD format at {squad_path}")
    
    print("All processing completed successfully!")

if __name__ == "__main__":
    main()