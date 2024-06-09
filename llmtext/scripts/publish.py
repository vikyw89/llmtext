def run():
    import subprocess

    subprocess.run(args="poetry version patch", shell=True)
    subprocess.run(args="poetry build", shell=True)
    subprocess.run(args="poetry publish", shell=True)