def run():
    import subprocess

    subprocess.run(args="pytest -v -rP -x", shell=True)
