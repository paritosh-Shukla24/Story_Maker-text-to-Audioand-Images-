from moviepy.editor import VideoFileClip

# Load your video
video_path = "C:\\Users\\ashutosh\\OneDrive\\Desktop\\stremlit\\WhatsApp Video 2024-03-07 at 12.45.21_b51ae301.mp4"
clip = VideoFileClip(video_path)

# Play the video
clip.preview()
