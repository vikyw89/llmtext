def run():
    import subprocess

    subprocess.run(args="poetry version patch", shell=True)