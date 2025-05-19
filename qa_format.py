import json
import os
import re

# Set the base directory
BASE_DIR = "/Users/casey/Documents/GitHub/LLM_Healthcare/output"
OUTPUT_DIR = os.path.join(BASE_DIR, "output/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_patient_records():
    """Load the existing patient_records.json file"""
    records_path = os.path.join(BASE_DIR, "patient_records.json")
    
    try:
        with open(records_path, 'r') as f:
            records = json.load(f)
        print(f"Successfully loaded {len(records)} records from patient_records.json")
        return records
    except Exception as e:
        print(f"Error loading patient_records.json: {e}")
        return []

def extract_patient_details(text):
    """Extract key details about the patient from the text"""
    details = {}
    
    # Extract sex
    sex_match = re.search(r'Sex:\s*([MF])', text)
    if sex_match:
        details['sex'] = 'Male' if sex_match.group(1) == 'M' else 'Female'
    
    # Extract age
    age_match = re.search(r'([0-9]+)-year-old\s+(fe)?male', text, re.IGNORECASE)
    if age_match:
        details['age'] = age_match.group(1)
    
    # Extract chief complaint
    chief_complaint_match = re.search(r'Chief\s+Complaint:\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
    if chief_complaint_match:
        details['chief_complaint'] = chief_complaint_match.group(1).strip()
    
    # Extract admission and discharge dates
    admission_date_match = re.search(r'Admission\s+Date:\s*\[\*\*(.*?)\*\*\]', text)
    discharge_date_match = re.search(r'Discharge\s+Date:\s*\[\*\*(.*?)\*\*\]', text)
    
    if admission_date_match and discharge_date_match:
        details['admission_date'] = admission_date_match.group(1)
        details['discharge_date'] = discharge_date_match.group(1)
        
        # Calculate length of stay if possible
        try:
            # This is just a placeholder since dates are deidentified
            details['length_of_stay'] = "Multiple days"
        except:
            pass
    
    # Extract surgical procedures
    surgical_procedure_match = re.search(r'Major\s+Surgical\s+or\s+Invasive\s+Procedure:\s*(.*?)(?:\n\n|\n[A-Z])', text, re.DOTALL | re.IGNORECASE)
    if surgical_procedure_match:
        details['surgical_procedures'] = surgical_procedure_match.group(1).strip()
    
    # Extract vital signs
    vitals_match = re.search(r'Vital\s+signs:?\s*(.*?)(?:\n\n|\n[A-Z])', text, re.DOTALL | re.IGNORECASE)
    if vitals_match:
        details['vital_signs'] = vitals_match.group(1).strip()
    
    # Extract hospital course summary - first paragraph of hospital course section
    hospital_course_match = re.search(r'HOSPITAL\s+COURSE:\s*(.*?)(?:\n\n|\n[A-Z])', text, re.DOTALL | re.IGNORECASE)
    if hospital_course_match:
        hospital_course = hospital_course_match.group(1).strip()
        # Take first 200 characters as a summary
        if len(hospital_course) > 200:
            details['hospital_course_summary'] = hospital_course[:200] + "..."
        else:
            details['hospital_course_summary'] = hospital_course
    
    # Extract discharge disposition
    disposition_match = re.search(r'(?:DISPOSITION:|Discharge\s+Disposition:)\s*(.*?)(?:\n\n|\n[A-Z]|$)', text, re.DOTALL | re.IGNORECASE)
    if disposition_match:
        details['disposition'] = disposition_match.group(1).strip()
    
    return details

def generate_qa_pairs(records):
    """Generate QA pairs from the records"""
    all_qa_pairs = []
    qa_pairs_by_record = []
    
    for i, record in enumerate(records):
        record_id = record.get("id", "unknown")
        text = record.get("text", "")
        medications = record.get("medications", [])
        diagnoses = record.get("diagnoses", [])
        
        # Extract patient details
        details = extract_patient_details(text)
        
        record_qa_pairs = []
        
        # Generate contextual questions about the patient
        if details.get('sex'):
            record_qa_pairs.append({
                "question": "What is the patient's sex?",
                "context": text,
                "answer": details['sex']
            })
        
        if details.get('age'):
            record_qa_pairs.append({
                "question": "How old is the patient?",
                "context": text,
                "answer": f"{details['age']} years old"
            })
        
        if details.get('chief_complaint'):
            record_qa_pairs.append({
                "question": "What was the patient's chief complaint?",
                "context": text,
                "answer": details['chief_complaint']
            })
        
        if details.get('surgical_procedures'):
            record_qa_pairs.append({
                "question": "What surgical procedures did the patient undergo?",
                "context": text,
                "answer": details['surgical_procedures']
            })
        
        if details.get('hospital_course_summary'):
            record_qa_pairs.append({
                "question": "Summarize the patient's hospital course.",
                "context": text,
                "answer": details['hospital_course_summary']
            })
        
        if details.get('disposition'):
            record_qa_pairs.append({
                "question": "What was the discharge disposition for this patient?",
                "context": text,
                "answer": details['disposition']
            })
        
        if details.get('vital_signs'):
            record_qa_pairs.append({
                "question": "What were the patient's vital signs?",
                "context": text,
                "answer": details['vital_signs']
            })
        
        if details.get('length_of_stay'):
            record_qa_pairs.append({
                "question": "How long was the patient hospitalized?",
                "context": text,
                "answer": details['length_of_stay']
            })
        
        # Add a general summary question combining key details
        summary_parts = []
        if details.get('age') and details.get('sex'):
            summary_parts.append(f"a {details['age']}-year-old {details['sex'].lower()}")
        if details.get('chief_complaint'):
            summary_parts.append(f"who presented with {details['chief_complaint'].lower()}")
        if diagnoses:
            summary_parts.append(f"and was diagnosed with {diagnoses[0]}")
        
        if summary_parts:
            summary = "This patient is " + ", ".join(summary_parts) + "."
            record_qa_pairs.append({
                "question": "Provide a brief summary of this patient.",
                "context": text,
                "answer": summary
            })
        
        # Generate medication questions
        if medications:
            # Question about all medications
            med_names = [med.get("name", "") for med in medications]
            
            # Question about all medications
            record_qa_pairs.append({
                "question": "What medications was the patient discharged with?",
                "context": text,
                "answer": ", ".join(med_names)
            })
            
            # Questions about specific medications (limit to 5 to avoid too many)
            for med in medications[:5]:
                med_name = med.get("name", "")
                med_dosage = med.get("dosage", "")
                
                if med_name and med_dosage:
                    # Question about medication dosage
                    record_qa_pairs.append({
                        "question": f"What is the dosage of {med_name}?",
                        "context": text,
                        "answer": f"{med_dosage} mg"
                    })
        
        # Generate diagnosis questions
        if diagnoses:
            # Question about all diagnoses
            record_qa_pairs.append({
                "question": "What were the patient's diagnoses at discharge?",
                "context": text,
                "answer": ", ".join(diagnoses)
            })
        
        # Store this record's QA pairs
        qa_pairs_by_record.append(record_qa_pairs)
        all_qa_pairs.extend(record_qa_pairs)
    
    print(f"\nTotal QA pairs generated: {len(all_qa_pairs)}")
    return all_qa_pairs, qa_pairs_by_record

def save_qa_pairs_to_text(qa_pairs_by_record):
    """Save QA pairs to a text file in the specified format"""
    output_path = os.path.join(OUTPUT_DIR, "all_qa_pairs.txt")
    
    with open(output_path, 'w') as f:
        for i, record_qa_pairs in enumerate(qa_pairs_by_record):
            f.write(f"QA PAIRS FOR RECORD {i+1}:\n")
            
            for j, qa in enumerate(record_qa_pairs):
                f.write(f"  {j+1}. Q: {qa['question']}\n")
                f.write(f"     A: {qa['answer']}\n")
                
            f.write("\n")
    
    print(f"Saved all QA pairs by record to {output_path}")

def create_qa_format(qa_pairs):
    """Create the simple QA format"""
    # Filter out empty or invalid pairs
    valid_pairs = [qa for qa in qa_pairs if qa.get("question") and qa.get("answer")]
    
    # Print a few examples of the final format
    print("\n=== Sample QA Pairs in Final Format ===")
    for i, qa in enumerate(valid_pairs[:3]):
        print(f"\nQA Pair {i+1}:")
        print(json.dumps(qa, indent=2))
    
    # Save to a file
    qa_path = os.path.join(OUTPUT_DIR, "qa_pairs_simple.json")
    with open(qa_path, "w") as f:
        json.dump(valid_pairs, f, indent=2)
    
    print(f"\nSaved {len(valid_pairs)} QA pairs to {qa_path}")
    return valid_pairs

def main():
    # Load the records
    records = load_patient_records()
    
    if not records:
        print("No records to process. Exiting.")
        return
    
    # Generate QA pairs
    all_qa_pairs, qa_pairs_by_record = generate_qa_pairs(records)
    
    # Save QA pairs to text file in the desired format
    save_qa_pairs_to_text(qa_pairs_by_record)
    
    # Create and save the QA format
    create_qa_format(all_qa_pairs)
    
    print("All processing completed successfully!")

if __name__ == "__main__":
    main()