import cProfile
import pstats
import io
from main import main_function  # Replace with the actual entry point of your application

def profile():
    profiler = cProfile.Profile()
    profiler.enable()
    main_function()  # Replace with the actual function to profile
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()

    with open("profile_output.txt", "w") as f:
        f.write(stream.getvalue())

if __name__ == "__main__":
    profile()
