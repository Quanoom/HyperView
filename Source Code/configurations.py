file_path = "NMF.py"

with open(file_path, "r") as file:
    lines = file.readlines()

# Remove mixed indentation and ensure 4-space indentation
fixed_lines = []
for line in lines:
    stripped_line = line.lstrip()  # Remove leading spaces/tabs
    fixed_line = "    " + stripped_line if stripped_line.startswith("print") else line.replace("\t", "    ")
    fixed_lines.append(fixed_line)

with open(file_path, "w") as file:
    file.writelines(fixed_lines)

print("✅ Fixed indentation issues in NMF.py!")
