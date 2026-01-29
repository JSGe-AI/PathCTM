import os
import re
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_txt_file(file_path):

    time_steps = []
    confidences = []
    

    pattern = re.compile(r"Time Step\s+(\d+):\s+Confidence:\s+([\d.]+)")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    time_steps.append(int(match.group(1)))
                    confidences.append(float(match.group(2)))
    except IOError as e:
        print(f"err0 {file_path} : {e}")
        return [], []
        
    return time_steps, confidences

def create_and_save_plot(time_steps, confidences, output_path, base_filename):

    if not time_steps or not confidences:
        print(f" {base_filename}.txt ")
        return


    plt.figure(figsize=(12, 7))


    plt.plot(time_steps, confidences, marker='o', linestyle='-', color='b', label='Confidence')


    plt.title(f'Confidence Trend for {base_filename}', fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Confidence', fontsize=12)

   
    plt.ylim(0, 1.05)

    plt.xlim(left=0)

 
    plt.grid(True, linestyle='--', alpha=0.6)

  
    plt.legend()
    
  
    plt.tight_layout()


    save_path = os.path.join(output_path, f"{base_filename}.png")
    
    try:
        plt.savefig(save_path, dpi=150) 
    except Exception as e:
        print(f" {save_path}: {e}")
    
 
    plt.close()


def main():

    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True,
        help="path"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help=""
    )
    args = parser.parse_args()

  
    if not os.path.isdir(args.input_dir):
        print(f" '{args.input_dir}' ")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{args.input_dir}")
    print(f" {args.output_dir}")


    try:
        txt_files = [f for f in os.listdir(args.input_dir) if f.endswith('.txt')]
        if not txt_files:
            print(" .txt ")
            return
    except OSError as e:
        print(f" '{args.input_dir}': {e}")
        return


    for filename in tqdm(txt_files, desc=""):
    
        file_path = os.path.join(args.input_dir, filename)
        
 
        time_steps, confidences = parse_txt_file(file_path)
        
   
        base_name = os.path.splitext(filename)[0]
        

        create_and_save_plot(time_steps, confidences, args.output_dir, base_name)

    print("\n end")


if __name__ == "__main__":
    main()