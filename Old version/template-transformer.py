#!/usr/bin/env python
import argparse
import os
import csv
import json
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='Transform i2b2 template data to JSON question-answer format.')
    parser.add_argument('--templates_dir', required=True, help='Path to the templates CSV file')
    parser.add_argument('--output_dir', required=True, help='Directory to output transformed JSON')
    return parser.parse_args()

def process_templates(templates_path, output_dir):
    """
    Process the templates CSV file and transform to JSON question-answer format.
    
    Args:
        templates_path (str): Path to the templates CSV file
        output_dir (str): Path to save the output JSON
    """
    print(f"Processing templates from {templates_path}")
    
    # Check if file exists
    if not os.path.exists(templates_path):
        print(f"Error: Templates file {templates_path} does not exist")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the templates CSV file
    templates = []
    with open(templates_path, 'r', encoding='utf-8') as f:
        # Skip the header line
        lines = f.readlines()
        
        # Process each line
        current_row = {}
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a header line
            if line.startswith('dataset') or line.startswith('Input') or line.startswith('Question'):
                continue
                
            # Check if this is a new row starting with 'risk'
            if line.startswith('risk'):
                if current_row and 'dataset' in current_row:
                    templates.append(current_row)
                current_row = {'dataset': 'risk'}
                
                # Split the line by commas or tabs
                parts = re.split(r'\t|,', line)
                if len(parts) >= 3:
                    current_row['input'] = parts[1].strip()
                    current_row['question'] = parts[2].strip()
            elif 'dataset' in current_row:
                # This is a continuation of the current row
                if 'logical_forms' not in current_row:
                    current_row['logical_forms'] = line
                elif 'answer_concepts' not in current_row:
                    current_row['answer_concepts'] = line
        
        # Add the last row
        if current_row and 'dataset' in current_row:
            templates.append(current_row)
    
    print(f"Found {len(templates)} templates")
    
    # Transform to JSON format
    qa_pairs = []
    for template in templates:
        # Extract the input fields
        input_fields = [field.strip() for field in template.get('input', '').split(',')]
        
        # Replace placeholders in the question
        question = template.get('question', '')
        
        # Generate a fake context based on the question and logical forms
        # This would need to be replaced with actual context extraction from medical records
        context = generate_context(question, template.get('logical_forms', ''), input_fields)
        
        # Generate the answer based on the answer_concepts
        answer = generate_answer(template.get('answer_concepts', ''), context)
        
        # Create a JSON entry
        qa_pair = {
            "question": question.replace('|test|', 'blood glucose').replace('|date|', 'March 15, 2023'),
            "context": context,
            "answer": answer
        }
        qa_pairs.append(qa_pair)
    
    # Write the JSON output
    output_file = os.path.join(output_dir, "qa_pairs.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2)
    
    print(f"Transformed templates saved to {output_file}")
    
    # Also save as JSONLines for easier processing
    jsonl_file = os.path.join(output_dir, "qa_pairs.jsonl")
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for qa_pair in qa_pairs:
            f.write(json.dumps(qa_pair) + '\n')
    
    print(f"Also saved as JSONLines to {jsonl_file}")

def generate_context(question, logical_forms, input_fields):
    """
    Generate a synthetic context based on the question and logical forms.
    This is a placeholder function - in a real scenario, you would extract the actual context from medical records.
    
    Args:
        question (str): The question template
        logical_forms (str): The logical forms
        input_fields (list): The input fields
    
    Returns:
        str: A generated context
    """
    # Replace placeholders with example values
    context = "MEDICAL RECORD EXTRACT:\n"
    
    if '|test|' in question:
        if 'abnormal' in question:
            context += "Lab results from March 15, 2023 show blood glucose level of 180 mg/dL (reference range: 70-99 mg/dL).\n"
            context += "This is flagged as abnormal and higher than the reference range.\n"
        else:
            context += "Lab results from March 15, 2023 show blood glucose level of 85 mg/dL (reference range: 70-99 mg/dL).\n"
    
    if 'medication' in question.lower():
        context += "DISCHARGE SUMMARY:\n"
        context += "The patient was sent home with metoprolol 50mg daily, aspirin 81mg daily, and lisinopril 10mg daily.\n"
    
    if 'risk' in logical_forms.lower():
        context += "Patient has a history of type 2 diabetes and hypertension. "
        context += "Recent lab work indicates controlled blood pressure but elevated HbA1c levels. "
        context += "Assessment indicates moderate risk for cardiovascular complications.\n"
    
    return context

def generate_answer(answer_concepts, context):
    """
    Generate an answer based on the answer_concepts and context.
    This is a placeholder function - in a real scenario, you would extract the actual answer.
    
    Args:
        answer_concepts (str): The answer concepts
        context (str): The generated context
    
    Returns:
        str: A generated answer
    """
    if 'none' in answer_concepts.lower():
        return "No information available"
    
    if 'result_date' in answer_concepts:
        return "85 mg/dL on March 15, 2023"
    
    if 'results' in answer_concepts:
        if 'abnormal' in context.lower():
            return "180 mg/dL (abnormal, higher than reference range)"
        else:
            return "85 mg/dL (within normal range)"
    
    if 'medication' in context.lower():
        return "metoprolol 50mg daily, aspirin 81mg daily, and lisinopril 10mg daily"
    
    if 'result_value_time' in answer_concepts:
        return "Blood glucose: 85 mg/dL, Measured at: March 15, 2023"
    
    # Default
    return "Data not available in the medical record"

def main():
    args = parse_arguments()
    process_templates(args.templates_dir, args.output_dir)

if __name__ == "__main__":
    main()