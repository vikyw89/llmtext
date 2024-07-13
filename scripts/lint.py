def run():
    import subprocess

    subprocess.run(args="ruff check . --fix", shell=True)
    subprocess.run(args="black .", shell=True)
