import torch
from train import SimpleNN

# --- SIMULATING LOADING YOUR MODEL ---
model = SimpleNN()
model.load_state_dict(torch.load("simple_nn_model.pth")) # Uncomment this line to load your real file
model.eval()

# 2. Open the header file
with open("model_weights.h", "w") as f:
    f.write("#ifndef MODEL_WEIGHTS_H\n")
    f.write("#define MODEL_WEIGHTS_H\n\n")

    # 3. Iterate through weights and write them as C arrays
    for name, param in model.named_parameters():
        # Flatten the weights to a 1D array for easy C access
        data = param.detach().numpy().flatten()
        
        # Clean up name to be a valid C variable (replace . with _)
        c_name = name.replace(".", "_")
        
        f.write(f"// Shape: {list(param.shape)}\n")
        f.write(f"const float {c_name}[] = {{\n")
        
        # Write data comma-separated
        data_str = ", ".join(map(str, data))
        f.write(f"    {data_str}\n")
        f.write("};\n\n")
        
        # Also write the dimensions for convenience
        if "weight" in name:
            f.write(f"const int {c_name}_out = {param.shape[0]};\n")
            f.write(f"const int {c_name}_in = {param.shape[1]};\n\n")

    f.write("#endif // MODEL_WEIGHTS_H\n")

print("Export complete! Created model_weights.h")