def run():
    import subprocess

    subprocess.run(args="pytest -v -s -x", shell=True)