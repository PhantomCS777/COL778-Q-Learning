# Read all lines from dummy_out.txt and store them in a list
file_path = 'dummy_out.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

# Remove any trailing newline characters
lines = [float(line.strip()) for line in lines]
print(min(lines))
print(sum(lines)/ len(lines))



# # Write the first 100 numbers to dummy_input.txt, each on a new line
# file_path = 'dummy_input.txt'

# with open(file_path, 'w') as file:
#     for num in range(1, 1001):
#         file.write(f"{num}\n")