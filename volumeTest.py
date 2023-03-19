import subprocess
import time

def get_output_volume():
    output = subprocess.run(['osascript', '-e', 'output volume of (get volume settings)'], capture_output=True, text=True)
    return int(output.stdout.strip())

def set_output_volume(volume):
    subprocess.run(['osascript', '-e', f'set volume output volume {volume}'])

def change_volume(delta):
    current_volume = get_output_volume()
    new_volume = current_volume + delta
    if new_volume < 0:
        new_volume = 0
    elif new_volume > 100:
        new_volume = 100
    set_output_volume(new_volume)

def main():
    # Increase by 5%
    change_volume(25)

    # Wait for 3 seconds
    time.sleep(3)

    # Decrease volume by 5%
    change_volume(-25)

if __name__ == '__main__':
    main()