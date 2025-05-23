import cProfile
import pstats
import io
import os
from main import main_function  # Replace with the actual entry point of your application

def profile(output_file="profile_output.txt", sort_by="cumulative", lines=50):
    """
    Profile the main function and save the profiling results to a file.

    Args:
        output_file (str): The file to save the profiling results.
        sort_by (str): The sorting criteria for the stats (e.g., 'cumulative', 'time').
        lines (int): The number of lines to display in the stats.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        main_function()  # Replace with the actual function to profile
    finally:
        profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats(sort_by)
    stats.print_stats(lines)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(stream.getvalue())
    print(f"Profiling results saved to {output_file}")

if __name__ == "__main__":
    profile()
