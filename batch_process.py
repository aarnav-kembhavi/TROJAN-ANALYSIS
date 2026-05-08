
import os
import subprocess
import shutil

# Paths
INPUT_ROOT = "External_Benchmarks"
OPENCORES_ROOT = "OpenCores_Extracted"
TROJAN_ROOT = "Dataset/Trojan"
OUTPUT_ROOT = "Dataset/Dataset" 
TEMP_JSON = "temp_netlist.json"
CONVERTER = "convert_verilog_to_gnn.py"

def run_wsl_yosys(verilog_path, json_output):
    v_path_linux = verilog_path.replace("\\", "/")
    cmd = f'wsl yosys -p "read_verilog {v_path_linux}; prep; write_json {json_output}"'
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Yosys failed for {v_path_linux}: {e.stderr.decode()}")
        return False

def process_directory(input_dir, output_dir, is_trojan=False, prefix="EXT"):
    verilog_files = []
    # Only search in likely RTL folders to avoid testbenches
    search_subfolders = ["rtl", "src", "verilog", "arithmetic", "random_control"]
    
    for root, dirs, files in os.walk(input_dir):
        if any(s in root.lower() for s in search_subfolders):
            for f in files:
                if f.endswith(".v") and not f.endswith("_tb.v"):
                    verilog_files.append(os.path.join(root, f))
    
    if not verilog_files:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.endswith(".v") and not f.endswith("_tb.v"):
                    verilog_files.append(os.path.join(root, f))
    
    print(f"Found {len(verilog_files)} Verilog designs in {input_dir}. Processing...")
    
    count = 0
    for v_path in verilog_files:
        if is_trojan:
            circuit_id = os.path.basename(os.path.dirname(v_path))
            target_dir = os.path.join(input_dir, circuit_id)
        else:
            name = os.path.splitext(os.path.basename(v_path))[0].upper()
            circuit_id = f"{prefix}_{name}"
            target_dir = os.path.join(output_dir, circuit_id)
            
        print(f"  [{count+1}/{len(verilog_files)}] {circuit_id}...")
        
        if run_wsl_yosys(v_path, TEMP_JSON):
            try:
                subprocess.run(["python", CONVERTER, TEMP_JSON, target_dir], check=True, capture_output=True)
                count += 1
            except subprocess.CalledProcessError as e:
                print(f"    Conversion failed: {e.stderr.decode()}")
        
        if os.path.exists(TEMP_JSON):
            os.remove(TEMP_JSON)
    return count

def main():
    # 1. Process AIG/Verilog Benchmarks
    c1 = process_directory(INPUT_ROOT, OUTPUT_ROOT, prefix="EXT_AIG")
    # 2. Process OpenCores IPs
    c2 = process_directory(OPENCORES_ROOT, OUTPUT_ROOT, prefix="OC")
    # 3. Process Trojan Benchmarks (to sync features)
    c3 = process_directory(TROJAN_ROOT, TROJAN_ROOT, is_trojan=True)
    
    print(f"\nBatch processing complete.")
    print(f"  External Benchmarks: {c1}")
    print(f"  OpenCores: {c2}")
    print(f"  Trojans Synchronized: {c3}")

if __name__ == "__main__":
    main()
