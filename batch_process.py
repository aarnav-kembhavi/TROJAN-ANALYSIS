
import os
import subprocess
import shutil

# Professional Restructured Paths
INPUT_ROOT = "Hardware_Security_Dataset/Source_Golden_Verilog"
TROJAN_ROOT = "Hardware_Security_Dataset/Processed_Trojan_GNN"
OUTPUT_ROOT = "Hardware_Security_Dataset/Processed_Golden_GNN" 
TEMP_JSON = "temp_netlist.json"
CONVERTER = "convert_verilog_to_gnn.py"

def run_wsl_yosys(verilog_path, json_output):
    v_dir = os.path.dirname(verilog_path).replace("\\", "/")
    v_file = os.path.basename(verilog_path)
    cmd = f'wsl yosys -p "read_verilog -sv {v_dir}/*.v; prep -top {os.path.splitext(v_file)[0]}; write_json {json_output}"'
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        v_path_linux = verilog_path.replace("\\", "/")
        cmd_fallback = f'wsl yosys -p "read_verilog -sv {v_path_linux}; prep; write_json {json_output}"'
        try:
            subprocess.run(cmd_fallback, shell=True, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

def process_directory(input_dir, output_dir, is_trojan=False, prefix="EXT"):
    verilog_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".v")]
    print(f"Found {len(verilog_files)} Verilog designs in {input_dir}. Processing...")
    count = 0
    for v_path in verilog_files:
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
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
    c1 = process_directory(INPUT_ROOT, OUTPUT_ROOT, prefix="CTU")
    # For Trojans, we only sync if there are raw files in there. 
    # Usually we process them once.
    print(f"\nBatch processing complete. Curated Golden: {c1}")

if __name__ == "__main__":
    main()
