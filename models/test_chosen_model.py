import subprocess

scripts = [
    # "model_base_test0.py",
    # "model_base_test1.py",
    # "mobilenet_finetune0.py",
    "mobilenet_finetune1.py",
    # "mobilenetv2_finetune0.py",
    "mobilenetv2_finetune1.py",
]

for script in scripts:
    subprocess.run(["python", script])
