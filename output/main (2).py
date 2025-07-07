from subprocess import check_call, CalledProcessError
import sys
import os
import csv

# Navigate to your project root
os.chdir("/Users/casey/Documents/GitHub/LLM_Healthcare/")

# Create the generation directory and its subdirectories
os.makedirs("generation/i2b2_medications", exist_ok=True)
os.makedirs("generation/i2b2_relations", exist_ok=True)
os.makedirs("generation/i2b2_heart_disease_risk", exist_ok=True)
os.makedirs("generation/i2b2_smoking", exist_ok=True)
os.makedirs("generation/i2b2_obesity", exist_ok=True)
os.makedirs("generation/combine_data", exist_ok=True)

PYTHON = sys.executable

# Current working directory
cwd = os.getcwd()

#################################### set the full file paths ###############################################

# Make these paths absolute to ensure they're correct
i2b2_relations_challenge_directory = os.path.abspath(os.path.join(cwd, "i2b2/relations/"))
i2b2_medications_challenge_directory = os.path.join(cwd, "i2b2/medication/")
i2b2_heart_disease_risk_challenge_directory = os.path.join(cwd, "i2b2/heart-disease-risk/")
i2b2_obesity_challenge_directory = os.path.join(cwd, "i2b2/obesity/")
i2b2_smoking_challenge_directory = os.path.join(cwd, "smoking-answers.py")
i2b2_coreference_challeneg_directory = os.path.join(cwd, "i2b2/coreference")

templates_directory = os.path.join(cwd, "/Users/casey/Documents/GitHub/LLM_Healthcare/templates-all.csv")

# Check if the generation directory exists
generation_dir = os.path.join(cwd, "generation")
if not os.path.exists(generation_dir):
    print(f"ERROR: The generation directory does not exist at {generation_dir}")
    print("Please make sure you have the correct directory structure:")
    print("- generation/")
    print("  |- i2b2_medications/")
    print("     |- medication-answers.py")
    print("  |- i2b2_relations/")
    print("     |- relations-answers.py")
    print("  |- i2b2_heart_disease_risk/")
    print("     |- risk-answers.py")
    print("  |- i2b2_smoking/")
    print("     |- smoking-answers.py")
    print("  |- i2b2_obesity/")
    print("     |- obesity-answers.py")
    print("  |- combine_data/")
    print("     |- combine_answers.py")
    sys.exit(1)

#################################### make output directory if it does not already exist #########################

model_dir = "output/"
if not os.path.exists(os.path.join(cwd, model_dir)):
    os.makedirs(os.path.join(cwd, model_dir))

output_directory = os.path.join(cwd, model_dir)  ## you can modify this to change the output directory path ##

###########################################################################################################

matching_notes = os.path.join(output_directory, "matching_notes.csv")
if not os.path.exists(matching_notes):
    ofile = open(matching_notes, "w")
    filewriter = csv.writer(ofile, delimiter="\t")
    filewriter.writerow(["relation", "coreference"])
    ofile.close()
else:
    print("matching_notes file already exists")

match_file = open(matching_notes)
csvreader = csv.reader(match_file)
matching_files = list(csvreader)  #  relation, coreference
new_file = []
new_file.append(matching_files[0])
flag = 0
for file in matching_files[1:]:
    if i2b2_relations_challenge_directory in file[0]:
        flag = 1
        break
    new_file.append([os.path.join(i2b2_relations_challenge_directory, file[0]), os.path.join(i2b2_coreference_challeneg_directory, file[1])])

if flag == 0:
    ofile = open(matching_notes, "w")
    filewriter = csv.writer(ofile, delimiter="\t")

    for val in new_file:
        filewriter.writerow(val)

    ofile.close()

################################### run the generation scripts #######################################

# Function to verify file exists before running and execute command
def run_command(script_path, i2b2_dir, templates_dir, output_dir):
    full_script_path = os.path.join(cwd, script_path)
    
    if not os.path.exists(full_script_path):
        print(f"ERROR: Script not found: {full_script_path}")
        return False
    
    cmd = f"{PYTHON} {full_script_path} --i2b2_dir={i2b2_dir} --templates_dir={templates_dir} --output_dir={output_dir}"
    print(cmd)
    
    try:
        check_call(cmd, shell=True)
        return True
    except CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {cmd}")
        return False

# Define script paths
script_paths = [
    ("generation/i2b2_medications/medication-answers.py", i2b2_medications_challenge_directory),
    ("generation/i2b2_relations/relations-answers.py", i2b2_relations_challenge_directory),
    ("generation/i2b2_heart_disease_risk/risk-answers.py", i2b2_heart_disease_risk_challenge_directory),
    ("generation/i2b2_smoking/smoking-answers.py", i2b2_smoking_challenge_directory),
    ("generation/i2b2_obesity/obesity-answers.py", i2b2_obesity_challenge_directory)
]

# Run each script
failures = []
for script_path, i2b2_dir in script_paths:
    success = run_command(script_path, i2b2_dir, templates_directory, output_directory)
    if not success:
        failures.append(script_path)

if failures:
    print("\nThe following scripts failed to run:")
    for script in failures:
        print(f"- {script}")
    print("\nPlease make sure these scripts exist and are correctly placed in the directory structure.")
    sys.exit(1)

##################  combine all the output files and generate the output in normal format ####################

combine_script = os.path.join(cwd, "generation/combine_data/combine_answers.py")
if not os.path.exists(combine_script):
    print(f"ERROR: Combine script not found: {combine_script}")
    sys.exit(1)

cmd = f"{PYTHON} {combine_script} --output_dir={output_directory}"
print(cmd)
try:
    check_call(cmd, shell=True)
except CalledProcessError as e:
    print(f"Combine command failed with exit code {e.returncode}: {cmd}")
    sys.exit(1)

print("\nAll scripts completed successfully!")

#####################  convert normal output to squad format ##################################
# Uncomment when needed

######################### basic analysis of the dataset #######################################
'''
cmd = "{python}  evaluation/analysis.py".format(python=PYTHON)
print(cmd)
check_call(cmd, shell=True)
'''