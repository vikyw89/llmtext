def run():
    import subprocess

    subprocess.run(args="poetry publish --build", shell=True)
