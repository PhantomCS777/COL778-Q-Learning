from re import sub
import sys ,os 
import subprocess 
import time 


if __name__ == "__main__":

    part_a = "python tabular_q.py --iterations 100000 --output_folder rough_output/"
    part_b = "python tabular_q.py --iterations 100000 --output_folder rough_output/part_b/ --part b"
    part_c = "python tabular_q.py --iterations 100000 --output_folder rough_output/part_c/ --part c"
    part_d = "python tabular_q.py --iterations 100000 --output_folder rough_output/part_d/ --part d"
    part_e_a = "python tabular_q.py --iterations 100000 --output_folder rough_output/part_e/ --part e_a"
    part_e_b = "python tabular_q.py --iterations 100000 --output_folder rough_output/part_e/ --part e_b"


    
    # subprocess.run(part_a, shell=True)
    # subprocess.run(part_b, shell=True)
    # subprocess.run(part_c, shell=True)
    subprocess.run(part_d, shell=True)
    # subprocess.run(part_e_a, shell=True)
    # subprocess.run(part_e_b, shell=True)

    

    
