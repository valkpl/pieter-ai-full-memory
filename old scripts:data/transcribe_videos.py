import os
import uuid
import shutil
import time
import threading
import yt_dlp
import browser_cookie3
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# Set your OpenAI API key
client = OpenAI(api_key="***REMOVED***")  # Replace with your real key

# Create transcripts output folder if it doesn't exist
output_dir = "data/transcripts"
os.makedirs(output_dir, exist_ok=True)

# Read URLs from the cleaned file
with open('video_urls_cleaned.txt', 'r') as f:
    urls = list(set(line.strip() for line in f if line.strip()))

# Grab Chrome cookies and save to a cookie file (Netscape format)
cj = browser_cookie3.chrome()

cookie_file_path = "chrome_cookies.txt"
with open(cookie_file_path, "w") as f:
    f.write("# Netscape HTTP Cookie File\n")
    for cookie in cj:
        domain = cookie.domain
        flag = "TRUE" if cookie.domain_initial_dot else "FALSE"
        path = cookie.path
        secure = "TRUE" if cookie.secure else "FALSE"
        expiration = str(cookie.expires or 0)
        name = cookie.name
        value = cookie.value
        f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expiration}\t{name}\t{value}\n")

# yt-dlp options
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'temp_audio/%(title)s.%(ext)s',
    'quiet': True,
    'noplaylist': True,
    'cookiefile': cookie_file_path,
    'postprocessors': [],
    'http_headers': {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
    },
}

# Create temp folder for downloaded audio
os.makedirs('temp_audio', exist_ok=True)

# Lock for thread-safe printing
print_lock = threading.Lock()

def process_video(idx, url):
    for attempt in range(3):
        try:
            with print_lock:
                print(f"\n🔻 [{idx}/{len(urls)}] Attempt {attempt+1}: Downloading audio from: {url}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', str(uuid.uuid4()))
                # Dynamically find the downloaded file (whatever extension)
                audio_files = os.listdir('temp_audio')
                audio_path = next((f"temp_audio/{f}" for f in audio_files if title in f), None)

            if not audio_path or not os.path.exists(audio_path):
                with print_lock:
                    print(f"❌ Could not find audio for {title} (attempt {attempt+1})")
                time.sleep(2)  # Small wait and retry
                continue

            with print_lock:
                print(f"🧠 Transcribing: {title}")

            with open(audio_path, 'rb') as audio_file:
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                transcript = transcript_response.text

            safe_title = "".join(c if c.isalnum() or c in " -" else "_" for c in title)
            save_path = os.path.join(output_dir, f"{safe_title}.txt")
            with open(save_path, "w", encoding="utf-8") as f_out:
                f_out.write(transcript)

            with print_lock:
                print(f"✅ Saved transcript: {save_path}")

            break  # Success! Exit retry loop

        except Exception as e:
            with print_lock:
                print(f"❌ Error processing {url} (attempt {attempt+1}): {e}")
            time.sleep(2)
            if attempt == 2:
                with print_lock:
                    print(f"🚫 Skipping {url} after 3 failed attempts.")

# Multithreaded download + transcription
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_video, idx, url) for idx, url in enumerate(urls, start=1)]

# Wait for all to complete
for future in futures:
    future.result()

# Cleanup temp audio folder
if os.path.exists('temp_audio'):
    shutil.rmtree('temp_audio')

print("\n🎉 All done processing all videos!")
